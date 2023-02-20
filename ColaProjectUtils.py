import argparse


def cola_trans_to_ip_port(s):
    ip, port = s.split(":")
    port = int(port)
    return ip, port


def cola_get_conf(ip_self, ip_target):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        type=str,
                        default=ip_target,
                        help='target ip:port')
    parser.add_argument('-s',
                        type=str,
                        default=ip_self,
                        help='self ip:port')
    args = parser.parse_args()
    args.t = cola_trans_to_ip_port(args.t)
    args.s = cola_trans_to_ip_port(args.s)
    print("Target: ", args.t)
    print("Self: ", args.s)
    return args
