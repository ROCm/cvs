'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''
import importlib.metadata
import os
import pytest
import sys

# Add all additional cmd line arguments for the script
def pytest_addoption(parser):
    parser.addoption( "--cluster_file", action="store", required=True, help="Input file with all the details of the cluster, nodes, switches in JSON format" )
    parser.addoption( "--config_file", action="store", required=True, help="Input file with all configurations and parameters for tests in JSON format" )

def pytest_metadata(metadata):
    """Add CVS version metadata for both console output and HTML report."""

    # Get CVS version - try package metadata first, fallback to version.txt
    try:
        cvs_version = importlib.metadata.version('cvs')
    except importlib.metadata.PackageNotFoundError:
        # Fallback for development mode (running from cloned repo)
        try:
            version_file = os.path.join(os.path.dirname(__file__), "..", "version.txt")
            with open(version_file) as f:
                cvs_version = f.read().strip()
        except Exception as e:
            cvs_version = f"Unknown (Error: {e})"

    # Get command line arguments directly from sys.argv
    cluster_file = "Not specified"
    config_file = "Not specified"

    # Parse command line arguments to get our custom options
    for i, arg in enumerate(sys.argv):
        if arg == "--cluster_file" and i + 1 < len(sys.argv):
            cluster_file = sys.argv[i + 1]
        elif arg == "--config_file" and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]

    # Add custom metadata
    metadata['CVS version'] = cvs_version
    metadata['Cluster File'] = cluster_file
    metadata['Config File'] = config_file
