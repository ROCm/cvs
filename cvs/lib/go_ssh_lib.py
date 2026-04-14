"""
Go SSH Library Python Wrapper

This module provides a Python interface to the high-performance Go SSH library.
It's designed as a drop-in replacement for the existing parallel_ssh_lib.py.
"""

import ctypes
import json
import os
from typing import List, Dict, Any


class GoSSHLibrary:
    """Wrapper for Go SSH shared library"""
    
    def __init__(self, lib_path: str = None):
        if lib_path is None:
            # Default path relative to this file (try both installed and development locations)
            lib_path = os.path.join(os.path.dirname(__file__), 'libssh.so')
            if not os.path.exists(lib_path):
                lib_path = os.path.join(os.path.dirname(__file__), '../../../cvs-go/libssh.so')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Go SSH library not found at {lib_path}")
            
        # Load shared library
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.ssh_execute_parallel.argtypes = [ctypes.c_char_p]
        self.lib.ssh_execute_parallel.restype = ctypes.c_char_p
        
        self.lib.ssh_execute_parallel_per_host.argtypes = [ctypes.c_char_p]
        self.lib.ssh_execute_parallel_per_host.restype = ctypes.c_char_p
        
        self.lib.ssh_free_result.argtypes = [ctypes.c_char_p]
        self.lib.ssh_free_result.restype = None
    
    def execute_parallel(self, nodes: List[str], command: str, username: str, 
                        priv_key_path: str, timeout: int = 60, 
                        concurrency: int = 200) -> Dict[str, Any]:
        """
        Execute command on multiple nodes in parallel using Go backend
        
        Args:
            nodes: List of node IP addresses/hostnames
            command: Command to execute
            username: SSH username
            priv_key_path: Path to private key file
            timeout: SSH timeout in seconds
            concurrency: Maximum concurrent connections
            
        Returns:
            Dict containing results for each node
        """
        # Prepare request
        request = {
            "nodes": nodes,
            "command": command,
            "username": username,
            "priv_key_path": priv_key_path,
            "timeout": timeout,
            "concurrency": concurrency
        }
        
        # Convert to JSON and call Go library
        request_json = json.dumps(request).encode('utf-8')
        result_ptr = self.lib.ssh_execute_parallel(request_json)
        
        # Convert result back to Python
        result_json = ctypes.string_at(result_ptr).decode('utf-8')
        result = json.loads(result_json)
        
        # Note: Skipping manual memory free for now to avoid double-free issues
        # Go's garbage collector will handle this
        
        return result
    
    def execute_parallel_per_host(self, node_commands: List[Dict[str, str]], username: str, 
                                 priv_key_path: str, timeout: int = 30, concurrency: int = 200) -> Dict[str, Any]:
        """
        Execute different commands on different hosts in parallel.
        
        Args:
            node_commands: List of {"host": "hostname", "command": "cmd"} dictionaries
            username: SSH username
            priv_key_path: Path to SSH private key
            timeout: Command timeout in seconds
            concurrency: Maximum parallel connections
            
        Returns:
            Dict with results from Go backend
        """
        request = {
            "node_commands": node_commands,
            "username": username,
            "priv_key_path": priv_key_path,
            "timeout": timeout,
            "concurrency": concurrency
        }
        
        request_json = json.dumps(request).encode('utf-8')
        result_ptr = self.lib.ssh_execute_parallel_per_host(request_json)
        result_str = ctypes.string_at(result_ptr).decode('utf-8')
        result = json.loads(result_str)
        
        return result


class Pssh:
    """
    Drop-in replacement for the original Pssh class using Go backend.
    
    This class maintains the same API as the original parallel_ssh_lib.Pssh
    but uses the high-performance Go SSH library underneath.
    """
    
    def __init__(self, log, host_list, user=None, password=None, pkey='id_rsa', 
                 host_key_check=False, stop_on_errors=True):
        self.log = log
        self.host_list = host_list
        self.reachable_hosts = host_list
        self.user = user
        self.password = password
        self.pkey = pkey
        self.host_key_check = host_key_check
        self.stop_on_errors = stop_on_errors
        self.unreachable_hosts = []

        # Debug: Print constructor parameters to multiple outputs
        import sys
        debug_msg = f"🔍 Go SSH Constructor DEBUG: user='{self.user}', pkey='{self.pkey}', hosts={len(host_list)} nodes"
        print(debug_msg, file=sys.stderr)  # Print to stderr
        print(debug_msg)  # Print to stdout
        
        # Also log to file for absolute confirmation
        try:
            with open('/tmp/go_ssh_debug.log', 'a') as f:
                import datetime
                f.write(f"{datetime.datetime.now()}: {debug_msg}\n")
        except:
            pass
        
        # Initialize Go SSH library
        try:
            self.go_ssh = GoSSHLibrary()
        except FileNotFoundError as e:
            if self.log:
                self.log.error(f"Go SSH library not available: {e}")
            raise
        
        
        # Validate required parameters
        if not self.user:
            raise ValueError("Username is required")
        if self.password:
            raise NotImplementedError("Password authentication not supported in Go backend")
        if not os.path.exists(self.pkey):
            raise FileNotFoundError(f"Private key file not found: {self.pkey}")
    
    def exec(self, cmd, timeout=None, print_console=True):
        """
        Execute command on all reachable hosts
        
        Returns a dictionary of host as key and command output as values
        """
        # Critical debug: Always log exec() calls
        try:
            with open('/tmp/go_ssh_debug.log', 'a') as f:
                import datetime
                f.write(f"{datetime.datetime.now()}: EXEC called with cmd='{cmd[:100]}...' on {len(self.reachable_hosts)} hosts\n")
        except:
            pass
        
        print(f"🔍 EXEC DEBUG: Running '{cmd[:50]}...' on {len(self.reachable_hosts)} hosts")
        if self.log:
            if timeout is not None:
                self.log.debug(f"Executing command on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]: {cmd}")
            else:
                self.log.debug(f"Executing command on {len(self.reachable_hosts)} host(s): {cmd}")
            self.log.debug(f"Go SSH backend using: user={self.user}, key={self.pkey}")
        
        # Always print debug info for troubleshooting
        print(f'🔍 Go SSH backend DEBUG: user={self.user}, key={self.pkey}')
        
        if print_console:
            print(f'cmd = {cmd}')
        
        # Use Go backend for parallel execution
        go_timeout = timeout if timeout is not None else 30
        result = self.go_ssh.execute_parallel(
            nodes=self.reachable_hosts,
            command=cmd,
            username=self.user,
            priv_key_path=self.pkey,
            timeout=go_timeout,
            concurrency=200  # Use moderate concurrency that worked well in shell tests
        )
        
        # Critical debug: Log when Go call returns
        try:
            with open('/tmp/go_ssh_debug.log', 'a') as f:
                import datetime
                f.write(f"{datetime.datetime.now()}: EXEC RETURNED from Go with {len(result.get('results', []))} results\n")
        except:
            pass
        
        print(f"🔍 EXEC DEBUG: Go returned {len(result.get('results', []))} results")
        
        # Convert Go results to expected format
        cmd_output = {}
        failed_hosts = []
        
        for node_result in result.get('results', []):
            host = node_result['host']
            
            if print_console:
                print('#----------------------------------------------------------#')
                print(f'Host == {host} ==')
                print('#----------------------------------------------------------#')
                print(cmd)
            
            if node_result['success']:
                # Success case
                output = node_result.get('stdout', '')
                if print_console and output:
                    print(output.strip())
                cmd_output[host] = output
            else:
                # Failure case
                error_msg = node_result.get('error', 'Unknown error')
                stderr = node_result.get('stderr', '')
                combined_error = f"{error_msg}\n{stderr}".strip()
                
                if print_console:
                    print(combined_error)
                
                cmd_output[host] = combined_error
                failed_hosts.append(host)
        
        # Handle failures based on stop_on_errors setting
        if failed_hosts and not self.stop_on_errors:
            # Update unreachable hosts list for future operations
            for host in failed_hosts:
                if host not in self.unreachable_hosts:
                    self.unreachable_hosts.append(host)
                    if host in self.reachable_hosts:
                        self.reachable_hosts.remove(host)
                        if print_console:
                            print(f"Host {host} is unreachable, pruning from reachable hosts list.")
        
        # Log completion
        if self.log:
            for host in cmd_output.keys():
                self.log.debug(f"Command completed on {host}: {cmd}")
        
        return cmd_output
    
    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """
        Run different commands on different hosts in parallel using efficient Go backend
        """
        if len(cmd_list) != len(self.reachable_hosts):
            raise ValueError("Command list length must match host list length")
        
        # Critical debug: Log exec_cmd_list calls
        try:
            with open('/tmp/go_ssh_debug.log', 'a') as f:
                import datetime
                f.write(f"{datetime.datetime.now()}: EXEC_CMD_LIST called with {len(cmd_list)} commands on {len(self.reachable_hosts)} hosts\n")
        except:
            pass
        
        if self.log:
            if timeout is not None:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s) [timeout={timeout}s]")
            else:
                self.log.debug(f"Executing command list on {len(self.reachable_hosts)} host(s)")
        
        if print_console:
            print(cmd_list)
        
        print(f"🚀 EXEC_CMD_LIST: Running {len(cmd_list)} different commands on {len(self.reachable_hosts)} hosts in parallel")
        
        # Build node_commands list for Go backend
        node_commands = []
        for host, cmd in zip(self.reachable_hosts, cmd_list):
            node_commands.append({
                "host": host,
                "command": cmd
            })
        
        # Use efficient Go backend for parallel per-host execution
        go_timeout = timeout if timeout is not None else 30
        result = self.go_ssh.execute_parallel_per_host(
            node_commands=node_commands,
            username=self.user,
            priv_key_path=self.pkey,
            timeout=go_timeout,
            concurrency=200  # High concurrency for parallel execution
        )
        
        # Critical debug: Log when Go per-host call returns
        try:
            with open('/tmp/go_ssh_debug.log', 'a') as f:
                import datetime
                f.write(f"{datetime.datetime.now()}: EXEC_CMD_LIST RETURNED from Go with {len(result.get('results', []))} results\n")
        except:
            pass
        
        print(f"🚀 EXEC_CMD_LIST: Go returned {len(result.get('results', []))} results")
        
        # Convert Go results to expected format
        cmd_output = {}
        
        for node_result in result.get('results', []):
            host = node_result['host']
            
            if node_result['success']:
                cmd_output[host] = node_result.get('stdout', '')
            else:
                error_msg = node_result.get('error', 'Unknown error')
                stderr = node_result.get('stderr', '')
                cmd_output[host] = f"{error_msg}\n{stderr}".strip()
        
        # Log per-host command execution
        if self.log:
            for host, cmd in zip(self.reachable_hosts, cmd_list):
                self.log.debug(f"Command on {host}: {cmd}")
        
        return cmd_output
    
    def scp_file(self, local_file, remote_file, recurse=False):
        """SCP file transfer - not implemented in Go backend yet"""
        raise NotImplementedError("SCP file transfer not yet implemented in Go backend")
    
    def reboot_connections(self):
        """Reboot connections - sends reboot command"""
        return self.exec('reboot -f')
    
    def destroy_clients(self):
        """Clean up resources"""
        # Go backend handles cleanup automatically
        if self.log:
            self.log.debug("Go SSH backend cleanup completed")
    
    def prune_unreachable_nodes(self):
        """Remove unreachable nodes from the host list"""
        if hasattr(self, 'unreachable_hosts') and self.unreachable_hosts:
            original_count = len(self.host_list)
            self.host_list = [h for h in self.host_list if h not in self.unreachable_hosts]
            self.reachable_hosts = [h for h in self.reachable_hosts if h not in self.unreachable_hosts]
            
            if self.log:
                pruned_count = original_count - len(self.host_list)
                self.log.info(
                    f"Pruned {pruned_count} unreachable nodes from phdl. Now managing {len(self.host_list)} reachable nodes."
                )


# Maintain backward compatibility
def scp(src, dst, srcusername, srcpassword, dstusername=None, dstpassword=None):
    """
    Legacy SCP function - not implemented in Go backend yet
    """
    raise NotImplementedError("SCP function not yet implemented in Go backend")
