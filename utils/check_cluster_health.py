


import os
import sys
import re
import logging
import argparse
import time

log = logging.getLogger()


sys.path.insert( 0, '../lib' )


import parallel_ssh_lib
import verify_lib
import linux_utils
import rocm_plib







def general_health_checks( phdl, ):
    print('Verify General Health Checks')
    # Check PCIe Bus and Width
    verify_lib.verify_gpu_pcie_bus_width( phdl )
    # Check Dmesg for errors
    verify_lib.full_dmesg_scan( phdl )
    # Check Dmesg for AMD GPU driver errors
    verify_lib.verify_driver_errors( phdl )
    # Check for any link flap evidence
    verify_lib.verify_nic_link_flap( phdl )
    # Check for GPU PCIe errors from amd-smi commands
    verify_lib.verify_gpu_pcie_errors( phdl )
    # Verify PCIe status from Host OS side ..
    verify_lib.verify_host_lspci( phdl)



#Things to do
#Html table comparison of error counters ..
#check and add Pytorch, tensorflow error patterns
#Add NCCL errors
#NIC CLI check - niccli -i 1 pcie -counters
#Congestion checker



def init_nic_dict( phdl, nic_dict ):
    print('Initialize NIC dict')
    nic_dict  = {}
    nic_dict['rdma_stats'] = {}
    nic_dict['eth_stats'] = {}
    nic_dict['pcie_stats'] = {}
    nic_dict['vendor_stats'] = {}
    for host in phdl.host_list:
        nic_dict['rdma_stats'][host] = {}
        nic_dict['eth_stats'][host] = {}
        nic_dict['pcie_stats'][host] = {}




def main():
    parser = argparse.ArgumentParser(description="Check Cluster Health" )
    parser.add_argument("--hosts_file", required=True,
          help = "File name with list of IP address one per line")
    parser.add_argument("--username", required=True,
          help = "Username to ssh to the hosts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--password", help="Password for username")
    group.add_argument("--key_file", help="Private Keyfile for username")
    parser.add_argument("--iterations", type=int, default=2, \
          help="Number of iterations to run the checks" )
    parser.add_argument("--time_between_iters", type=int, default=60, \
          help="Time duration to sleep between iterations .." )
    parser.add_argument("--check_nic", type=bool, default=True, \
          help="Check NIC related metrics" )
    parser.add_argument("--check_gpu", type=bool, default=True, \
          help="Check gpu related metrics" )
    parser.add_argument("--check_dmesg", type=bool, default=True, \
          help="Scan dmesg for errors" )
    parser.add_argument("--report_file", default="./cluster_report.html" )
    args = parser.parse_args()

    # Read host IP addresses from input file.
    with open(args.hosts_file, "r") as f:
         node_list = [line.strip() for line in f if line.strip()]
    if not node_list:
        print("ERROR !! No hosts in the file, this is mandatory, aborting !!")
        sys.exit(1)
    if len(node_list) == 0:
        print("ERROR !! No hosts in the file, this is mandatory, aborting !!")
        sys.exit(1)


    # Connect to all hosts in the cluster ..
    if args.key_file:
        phdl = parallel_ssh_lib.Pssh( log, node_list, user=args.username, pkey=args.key_file )
    elif args.password:
        phdl = parallel_ssh_lib.Pssh( log, node_list, user=args.username, password=args.password )



    gen_results_dict = {}
    # Verify general health checks .. which should never be seen ..
    # PCIe Bus and Width ..
    # PCIe LTSSM Changes
    # Correctable errors, Uncorrectable errors
    # Fatal Interrupts ..
    # Link down
    # Link flaps in dmesg
    # PCIe errors
    # Link Errors
    #
    # To be done
    # XGMI Link status, metrics
    # RAS error checks
    # Journlctl scan for interrupts
    # Vendor specific stats
    # Add NIC MTU Check
    # Add PFC, ECN config checks
    #

    general_health_checks( phdl )



    # Take cluster metrics snapshot before iterations ..
    snapshot_dict_before = verify_lib.create_cluster_metrics_snapshot( phdl )


    snapshot_iters_dict = {}
    for i in range(1, int(args.iterations)+1):
        print('#------------------------------------------------------------#')
        print(f'Starting Iteration - {i}')
        print('#------------------------------------------------------------#')
        snapshot_iters_dict[i] = verify_lib.create_cluster_metrics_snapshot( phdl )
        time.sleep(int(args.time_between_iters))


    print('#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#')
    print('Completed all iterations, taking final snapshot for comparison')
    print('#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#')

    snapshot_dict_after = verify_lib.create_cluster_metrics_snapshot( phdl )
    verify_lib.compare_cluster_metrics_snapshots( snapshot_dict_before, snapshot_dict_after )

if __name__ == "__main__":
    main()



