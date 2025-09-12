'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''


#!/bin/bash

# Please change to your user-id
export HTML_DIR='/var/www/html/cvs/'
export LOG_DIR='/local/venksrin/cvs/'
export CLUSTER_FILE='/home/venksrin/cvs/input/cluster.json'
export CONFIG_FILE='/home/venksrin/cvs/input/mi300_config.json'

mkdir $HTML_DIR
mkdir $LOG_DIR

pytest -vvv --log-file=$LOG_DIR/agfhc_test.log -s agfhc_cvs.py --cluster_file $CLUSTER_FILE --config_file $CONFIG_FILE --html=$HTML_DIR/agfhc_test_report.html --capture=tee-sys --self-contained-html > $LOG_DIR/agfhc_script.log
pytest -vvv --log-file=$LOG_DIR/agfhc_test.log -s transferbench_cvs.py --cluster_file $CLUSTER_FILE --config_file $CONFIG_FILE --html=$HTML_DIR/transferbench_test_report.html --capture=tee-sys --self-contained-html > $LOG_DIR/transferbench_script.log
pytest -vvv --log-file=$LOG_DIR/agfhc_test.log -s rocblas_cvs.py --cluster_file $CLUSTER_FILE --config_file $CONFIG_FILE --html=$HTML_DIR/rocblas_test_report.html --capture=tee-sys --self-contained-html > $LOG_DIR/rocblas_script.log
pytest -vvv --log-file=$LOG_DIR/agfhc_test.log -s babelstream_cvs.py --cluster_file $CLUSTER_FILE --config_file $CONFIG_FILE --html=$HTML_DIR/babelstream_test_report.html --capture=tee-sys --self-contained-html > $LOG_DIR/babelstream_script.log
