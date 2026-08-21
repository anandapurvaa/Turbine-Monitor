import os
import argparse
from pathlib import Path
from datetime import datetime
from google.cloud import storage
import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import PatchTSTConfig, PatchTSTModel
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.ml.dataset import get_dataloaders
from src.ml.train_patchtst import PatchTSTRegressor

load_dotenv()


class PatchTSTRegressor(nn.Module):
    """
    Custom wrapper around PatchTST encoder for regression.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = PatchTSTModel(config)
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.d_model // 2, 1)
        )
        
    def forward(self, x):
        outputs = self.encoder(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=(1, 2))
        pred = self.regression_head(pooled)
        pred = pred.squeeze(-1)
        return pred


def download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download file from GCS to local temp directory"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")


def upload_model_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload trained model to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")


def retrain_and_upload():
    """Main retraining function"""
    
    # Configuration
    bucket_name = os.getenv("GCS_BUCKET", "turbineguard-data")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Paths
    data_dir = Path("temp_data")
    data_dir.mkdir(exist_ok=True)
    
    train_csv = data_dir / "train_FD001_processed.csv"
    test_csv = data_dir / "test_FD001_processed.csv"
    model_path = "best_model.pt"
    
    # Download training data from GCS
    print("Downloading training data from GCS...")
    download_data_from_gcs(bucket_name, "processed/train_FD001_processed.csv", str(train_csv))
    download_data_from_gcs(bucket_name, "processed/test_FD001_processed.csv", str(test_csv))
    
    # Initialize MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("turbineguard_patchtst_fd001")
    
    # Training parameters
    context_length = 36
    batch_size = 64
    epochs = 50
    lr = 3e-4
    weight_decay = 1e-4
    patience = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, test_loader, train_stats = get_dataloaders(
        train_csv,
        test_csv,
        context_length=context_length,
        batch_size=batch_size,
    )
    
    # Model config
    config = PatchTSTConfig(
        context_length=context_length,
        patch_length=16,
        patch_stride=8,
        num_input_channels=21,
        d_model=128,
        encoder_layers=3,
        encoder_attention_heads=4,
        encoder_ffn_dim=256,
        dropout=0.1,
        head_dropout=0.1,
    )
    
    model = PatchTSTRegressor(config).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    best_mae = float("inf")
    patience_counter = 0
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"retrain_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "context_length": context_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "d_model": config.d_model,
            "encoder_layers": config.encoder_layers,
            "num_input_channels": config.num_input_channels,
        })
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Eval
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())
            
            val_loss /= len(test_loader.dataset)
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            mae = np.mean(np.abs(all_preds - all_targets))
            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
            
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {mae:.2f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mae": mae,
                "rmse": rmse,
            }, step=epoch)
            
            scheduler.step(val_loss)
            
            if mae < best_mae:
                best_mae = mae
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "mae": mae,
                    "rmse": rmse,
                    "train_stats": train_stats,
                    "config": config.to_dict(),
                    "timestamp": timestamp,
                }
                torch.save(checkpoint, model_path)
                print(f"New best MAE: {mae:.2f} — model saved")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        mlflow.log_metric("best_mae", best_mae)
        
        # Upload model to GCS
        gcs_model_path = f"models/patchtst_fd001_{timestamp}.pt"
        upload_model_to_gcs(bucket_name, model_path, gcs_model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Training complete. Best MAE: {best_mae:.2f}")
        print(f"Model uploaded to GCS: {gcs_model_path}")
        
        return {"status": "success", "best_mae": best_mae, "model_path": gcs_model_path}


if __name__ == "__main__":
    result = retrain_and_upload()
    print(result)