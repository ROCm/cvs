'''Role and endpoint resolution for llm-d vLLM replicas.'''


class LlmdTopology:
    def __init__(self, config, cluster_dict):
        self.gateway_node = config.gateway.node
        self.workers = []
        node_dict = cluster_dict["node_dict"]
        for worker in config.workers:
            node = worker["node"]
            reachable_address = node_dict[node].get("vpc_ip") or node
            endpoint_address = "127.0.0.1" if node == self.gateway_node else reachable_address
            resolved = dict(worker)
            resolved["address"] = endpoint_address
            self.workers.append(resolved)

        self.hosts = []
        for host in [self.gateway_node] + [worker["node"] for worker in self.workers]:
            if host not in self.hosts:
                self.hosts.append(host)

    def workers_on(self, host):
        return [worker for worker in self.workers if worker["node"] == host]

    def worker_url(self, worker):
        return f"http://{worker['address']}:{worker['port']}"


def scope_cluster(cluster_dict, topology):
    '''Restrict SSH execution to llm-d role hosts and put the gateway first.'''
    scoped = dict(cluster_dict)
    scoped["orchestrator"] = "baremetal"
    scoped["node_dict"] = {host: cluster_dict["node_dict"][host] for host in topology.hosts}
    scoped["head_node_dict"] = {"mgmt_ip": topology.gateway_node}
    return scoped
