'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Add all additional cmd line arguments for the script
def pytest_addoption(parser):
    parser.addoption( "--cluster_file", action="store", required=True, help="Input file with all the details of the cluster, nodes, switches in JSON format" )
    parser.addoption( "--config_file", action="store", required=True, help="Input file with all configurations and parameters for tests in JSON format" )
