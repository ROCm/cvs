import re
import sys
import os
import json

from utils_lib import *






def get_lshw_network_dict( phdl ):
    lshw_dict = {}
    out_dict = phdl.exec('sudo lshw -class network -businfo')
    for node in out_dict.keys():
        lshw_dict[node] = {}
        for line in out_dict[node].split("\n"):
            print(line)
            if re.search( 'pci\@([0-9a-f\:\.]+)\s+([a-z0-9\-\.]+)\s+network\s+([a-z0-9\s\[\]\/\-\_]+)', line, re.I ):
                match = re.search( 'pci\@([0-9a-f\:\.]+)\s+([a-z0-9\-\.]+)\s+network\s+([a-z0-9\s\[\]\/\-\_]+)', line, re.I )
                pci_bus = match.group(1)
                dev_name = match.group(2)
                dev_descr = match.group(3)
                lshw_dict[node][dev_name] = {}
                lshw_dict[node][dev_name]['pci_bus'] = pci_bus
                lshw_dict[node][dev_name]['description'] = dev_descr
            elif re.search( 'pci\@([0-9a-f\:\.]+)\s+network\s+([a-z0-9\s\[\]\/\-\_]+)', line, re.I ):
                match = re.search( 'pci\@([0-9a-f\:\.]+)\s+network\s+([a-z0-9\s\[\]\/\-\_]+)', line, re.I )
                pci_bus = match.group(1)
                dev_name = 'virtio'
                dev_descr = match.group(2)
                lshw_dict[node][dev_name] = {}
                lshw_dict[node][dev_name]['pci_bus'] = pci_bus
                lshw_dict[node][dev_name]['description'] = dev_descr
                
    return lshw_dict
              
     

def get_ip_addr_dict( phdl ):
    ip_dict = {}
    out_dict = phdl.exec('sudo ip addr show | grep -A 5 mtu --color=never')
    int_nam = None
    for node in out_dict.keys():
        ip_dict[node] = {}
        for line in out_dict[node].split("\n"):
            if re.search( '[0-9]+\:\s+([0-9a-z\.\_\-\/]+):\s+([\<\>\,A-Z0-9]+)', line ):
                match = re.search( '[0-9]+\:\s+([0-9a-z\.\_\-\/]+):\s+([\<\>\,A-Z0-9]+)', line )
                int_nam = match.group(1)
                ip_dict[node][int_nam] = {}
                ip_dict[node][int_nam]['ipv4_addr_list'] = []
                ip_dict[node][int_nam]['ipv6_addr_list'] = []
                 
                ip_dict[node][int_nam]['flags'] = match.group(2)
            if re.search( 'mtu ([0-9]+)', line ):
                match = re.search( 'mtu ([0-9]+)', line )
                ip_dict[node][int_nam]['mtu'] = match.group(1)
            if re.search( 'state ([A-Z]+)', line ):
                match = re.search( 'state ([A-Z]+)', line )
                ip_dict[node][int_nam]['state'] = match.group(1)
            if re.search( 'link\/ether\s+([a-f0-9\:]+)', line ):
                match = re.search( 'link\/ether\s+([a-f0-9\:]+)', line )
                ip_dict[node][int_nam]['mac_addr'] = match.group(1)
            if re.search( 'inet\s+([0-9\.\/]+)', line ):
                match = re.search( 'inet\s+([0-9\.\/]+)', line )
                ip_dict[node][int_nam]['ipv4_addr_list'].append(match.group(1))
            if re.search( 'inet6\s+([a-f0-9\:\/]+)', line ):
                match = re.search( 'inet6\s+([a-f0-9\:\/]+)', line )
                ip_dict[node][int_nam]['ipv6_addr_list'].append(match.group(1))

    return ip_dict



def get_rdma_nic_dict( phdl ):
    rdma_dict = {}
    out_dict = phdl.exec('sudo rdma link')
    #gid_dict_t = phdl.exec('sudo show_gids | grep -i v2 --color=never')
    for node in out_dict.keys():
        rdma_dict[node] = {}
        for line in out_dict[node].split("\n"):
            if re.search( '^link', line ):
                match = re.search( 'link\s+([a-zA-Z0-9\_\-\.]+)\/([0-9]+)\s+state\s+([A-Za-z]+)\s+physical_state\s+([A-Za-z\_]+)\s+netdev\s+([a-z0-9A-Z\.]+)', line)
                dev = match.group(1)
                rdma_dict[node][dev] = {}
                rdma_dict[node][dev]['port'] = match.group(2)
                rdma_dict[node][dev]['device_status'] = match.group(3)
                rdma_dict[node][dev]['link_status'] = match.group(4)
                rdma_dict[node][dev]['eth_device'] = match.group(5)
    return rdma_dict

            
   

def get_backend_nic_dict( phdl ):
    lshw_dict = get_lshw_network_dict( phdl )
    bck_net_dict = {}
    for node in lshw_dict.keys():
        bck_net_dict[node] = []
        print(lshw_dict[node])
        # we make a crude assumption that number of backend nics are more than front end ..
        # if we have to pass this from Input file, we can do that.
        list_a = []
        list_b = []
        for intf_name in lshw_dict[node].keys():
            if len(list_a) == 0:
                list_a.append(intf_name)
            else:
                if lshw_dict[node][intf_name]['description'] == lshw_dict[node][list_a[0]]['description']:
                    list_a.append(intf_name)
                else:
                    list_b.append(intf_name)
        if len(list_a) > len(list_b):
            bck_net_dict[node] = list_a
        else:
            bck_net_dict[node] = list_b
    return bck_net_dict
    

def get_backend_rdma_nic_dict( phdl ):
    bck_rdma_nic_dict = {}
    bck_nic_dict = get_backend_nic_dict( phdl )
    rdma_nic_dict = get_rdma_nic_dict( phdl )
    for node in rdma_nic_dict.keys():
        bck_rdma_nic_dict[node] = {}
        bck_nic_list = bck_nic_dict[node]
        for rdma_dev in rdma_nic_dict[node].keys():
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_list:
                bck_rdma_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]
    return bck_rdma_nic_dict


def convert_ethtool_out_to_dict( ethtool_out, vendor=None ):
    out_dict = {}
    # For now, let us ignore the per queue stats and just collect total stats
    match_list = re.findall( '([a-z\_\-]+\:\s+[0-9]+)', ethtool_out, re.I )
    print(match_list)
    for match_item in match_list:
        match = re.search( '([a-z\_\-]+)\:\s+([0-9]+)', match_item, re.I )
        out_dict[match.group(1)] = match.group(2)
    return out_dict



# stats_dict will be indexed by node_ip, followed by interface name of backend NICs
 
def get_nic_ethtool_stats_dict( phdl, vendor=None ):
    stats_dict = {}
    bck_nic_dict = get_backend_nic_dict( phdl )
    node_list = list(bck_nic_dict.keys())
    node_0 = node_list[0]
    no_of_nics = len(bck_nic_dict[node_0])
    #print(no_of_nics)
    # initialize stats dict
    for node in node_list:
        stats_dict[node] = {}

    # Build a list of list of cmds with the assumptions that NICs can be with different 
    # interface names across nodes and we still want to do parallel execution ..
    # cmd_dict is a dict with key as nodes and value as list of cmds
    cmd_dict = {}
    print(node_list)
    print(bck_nic_dict)
    for i in range(0,no_of_nics):
        cmd_dict[i] = []
        for node in node_list:
            intf_nam = bck_nic_dict[node][i]
            cmd_dict[i].append( f'sudo ethtool -S {intf_nam}' )
            stats_dict[node][intf_nam] = {}
    for i in range(0,no_of_nics):
        cmd_list = cmd_dict[i]
        stats_dict_out = phdl.exec_cmd_list(cmd_list)
        for node in stats_dict_out:
            intf_nam = bck_nic_dict[node][i]
            stats_dict[node][intf_nam] = convert_ethtool_out_to_dict(stats_dict_out[node], vendor )

    for node in stats_dict.keys():
        for intf in stats_dict[node].keys():
            for counter in stats_dict[node][intf].keys():
                if re.search( 'err|discard|drop|crc|fcs|reset', counter, re.I ):
                    if int(stats_dict[node][intf][counter]) > 0:
                        print(f'WARN !! {node} {intf} {counter} {stats_dict[node][intf][counter]}')
    return stats_dict
            



def get_lldp_dict( phdl ):
    lldp_dict = {}
    print('Get LLDP dict')
    out_dict = phdl.exec('sudo lldpcli show neighbors -f json')
    for node in out_dict.keys():
        lldp_dict[node] = json_to_dict(out_dict[node])     
    return lldp_dict


def get_dns_dict( phdl ):
    dns_dict = {}
    out_dict = phdl.exec('sudo resolvectl status | head -7')
    for node in out_dict.keys():
        dns_dict[node] = {}
        for line in out_dict[node].split("\n"):
            if re.search( 'Protocols', line, re.I ):
                print('')
            elif re.search( 'Protocols', line, re.I ):
                print('')
            elif re.search( 'Current DNS Server', line, re.I ):
                print('')
            elif re.search( 'DNS Servers', line, re.I ):
                print('')
            elif re.search( 'DNS Domain', line, re.I ):
                print('')
    return dns_dict


def get_rdma_stats_dict( phdl ):

    rdma_stats_dict = {}
    bck_nic_dict = get_backend_rdma_nic_dict(phdl)
    out_dict = phdl.exec('sudo rdma statistic --json')
    for node in out_dict.keys():
        bck_nic_list = bck_nic_dict[node]
        print(bck_nic_list)
        rdma_stats_dict[node] = {}
        rdma_dict_list = json_to_dict(out_dict[node]) 
        for rdma_dict in rdma_dict_list:
            device_name = rdma_dict['ifname']
            if device_name in bck_nic_list:
                rdma_stats_dict[node][device_name] = rdma_dict
    return rdma_stats_dict





def get_linux_perf_tuning_dict( phdl ):
    out_dict = {}
    out_dict['bios_version'] = phdl.exec('sudo dmidecode -s bios-version')
    out_dict['numa_balancing'] = phdl.exec('sudo sysctl kernel.numa_balancing')
    out_dict['nmi_watchdog'] = phdl.exec('sudo cat /proc/sys/kernel/nmi_watchdog')
    out_dict['huge_pages'] = phdl.exec('sudo cat /sys/kernel/mm/transparent_hugepage/enabled')
    out_dict['cpu_power_profile'] = phdl.exec('sudo cpupower info')


