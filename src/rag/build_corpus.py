from pathlib import Path
from typing import List, Dict
import json

# Synthetic maintenance manuals for C-MAPSS failure modes
# Each manual is tied to specific sensor signatures from the dataset

MANUALS = [
    {
        "id": "HPC-001",
        "title": "High Pressure Compressor Efficiency Degradation",
        "content": """
        High Pressure Compressor (HPC) efficiency degradation is characterized by:
        - Increased HPC outlet temperature (sensor_14)
        - Decreased HPC pressure ratio (sensor_15)
        - Reduced fan speed (sensor_7) as the engine compensates
        - Elevated total temperature at HPC outlet (sensor_11)
        
        This is the primary failure mode in the FD001 dataset. The degradation
        progresses gradually over 150-200 cycles before reaching critical levels.
        
        Maintenance action: Schedule HPC blade inspection and cleaning. Replace
        if efficiency drop exceeds 5% from baseline.
        """,
        "sensor_signatures": ["sensor_14", "sensor_15", "sensor_7", "sensor_11"],
    },
    {
        "id": "FAN-001",
        "title": "Fan Speed Sensor Drift and Calibration Error",
        "content": """
        Fan speed sensor drift manifests as:
        - Gradual deviation in fan speed readings (sensor_7)
        - Discrepancy between commanded and actual fan speed
        - May correlate with changes in bypass duct flow (sensor_5)
        
        This failure mode appears in FD003 and FD004 datasets alongside HPC
        degradation.
        
        Maintenance action: Recalibrate fan speed sensor. Verify with ground
        truth measurements.
        """,
        "sensor_signatures": ["sensor_7", "sensor_5"],
    },
    {
        "id": "TEMP-001",
        "title": "Excessive Core Temperature Rise",
        "content": """
        Abnormal core temperature patterns include:
        - Rising core temperature (sensor_12)
        - Elevated exhaust gas temperature (sensor_17)
        - Increased fuel flow to maintain thrust (sensor_8)
        
        Often accompanies HPC degradation but can occur independently due to
        cooling system issues.
        
        Maintenance action: Inspect cooling passages, check for blockages.
        """,
        "sensor_signatures": ["sensor_12", "sensor_17", "sensor_8"],
    },
    {
        "id": "PRESS-001",
        "title": "Pressure Anomaly in Bypass Duct",
        "content": """
        Bypass duct pressure anomalies show:
        - Fluctuating bypass duct pressure (sensor_5)
        - Correlated changes in total pressure at HPC outlet (sensor_16)
        - Possible fan blade vibration indicators
        
        May indicate foreign object damage or buildup in bypass duct.
        
        Maintenance action: Visual inspection of bypass duct, clean if needed.
        """,
        "sensor_signatures": ["sensor_5", "sensor_16"],
    },
    {
        "id": "OIL-001",
        "title": "Lubrication System Degradation",
        "content": """
        Lubrication issues manifest through:
        - Rising oil temperature indicators (sensor_6)
        - Changes in hydraulic pressure patterns
        - Increased friction-related temperature rise
        
        Secondary effect: can accelerate bearing wear and compressor degradation.
        
        Maintenance action: Check oil quality, replace if contaminated. Inspect
        bearings for wear.
        """,
        "sensor_signatures": ["sensor_6"],
    },
    {
        "id": "BLEED-001",
        "title": "Bleed Air System Malfunction",
        "content": """
        Bleed air system issues show:
        - Anomalous bleed air pressure readings
        - Changes in compressor operating line
        - Possible surge margin reduction
        
        Can affect engine stability and fuel efficiency.
        
        Maintenance action: Inspect bleed valves, check for leaks.
        """,
        "sensor_signatures": ["sensor_15", "sensor_16"],
    },
    {
        "id": "CONTROL-001",
        "title": "Engine Control System Response Anomaly",
        "content": """
        Control system response issues:
        - Sluggish throttle response (visible in sensor_1, sensor_2, sensor_3)
        - Delayed actuator movements
        - Possible sensor feedback loop issues
        
        May be software-related or hardware degradation.
        
        Maintenance action: Run control system diagnostics, update software if
        available.
        """,
        "sensor_signatures": ["sensor_1", "sensor_2", "sensor_3"],
    },
    {
        "id": "VIB-001",
        "title": "Vibration and Imbalance Indicators",
        "content": """
        Vibration-related failure signatures:
        - Oscillating patterns in multiple sensors
        - Correlated changes in fan speed and pressure readings
        - Possible bearing wear indicators
        
        Early detection critical to prevent catastrophic failure.
        
        Maintenance action: Vibration analysis, balance check, bearing inspection.
        """,
        "sensor_signatures": ["sensor_7", "sensor_15", "sensor_16"],
    },
    {
        "id": "DEGRADE-001",
        "title": "General Performance Degradation Pattern",
        "content": """
        Overall engine health decline shows:
        - Gradual reduction in maximum achievable thrust
        - Increased fuel consumption for same operating point
        - Shift in operating envelope boundaries
        
        This is a catch-all category for non-specific degradation.
        
        Maintenance action: Comprehensive engine health assessment, consider
        overhaul if multiple systems degraded.
        """,
        "sensor_signatures": ["sensor_8", "sensor_14", "sensor_15"],
    },
    {
        "id": "SENSOR-001",
        "title": "Sensor Calibration Drift",
        "content": """
        Sensor drift patterns:
        - Gradual bias in one or more sensor readings
        - Inconsistency between redundant measurements
        - Sudden jumps after maintenance events
        
        Can mimic actual component degradation if not identified.
        
        Maintenance action: Cross-check with ground truth, recalibrate affected
        sensors.
        """,
        "sensor_signatures": ["sensor_11", "sensor_12", "sensor_14"],
    },
]


def build_corpus(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual manual files
    for manual in MANUALS:
        manual_path = output_dir / f"{manual['id']}.md"
        with open(manual_path, "w", encoding="utf-8") as f:
            f.write(f"# {manual['title']}\n\n")
            f.write(f"**ID:** {manual['id']}\n\n")
            f.write(manual['content'])
    
    # Save combined corpus JSON
    corpus_path = output_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(MANUALS, f, indent=2)
    
    print(f"Created {len(MANUALS)} manual files in {output_dir}")
    print(f"Corpus saved to {corpus_path}")
    
    return MANUALS


if __name__ == "__main__":
    build_corpus(Path("data/manuals"))