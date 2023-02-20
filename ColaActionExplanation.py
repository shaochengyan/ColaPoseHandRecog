import yaml


def load_conf(filename):
    fin = open(filename, "r")
    fin.seek(0)  # 回到文件首
    data = yaml.safe_load(fin)
    return data


class ColaActionExplanation:

    def __init__(self, conf_filename="./config.yaml") -> None:
        self.action_name_dict = load_conf(conf_filename)
        # print(self.action_name_dict)

    def explain(self, action_dict):
        for key, value in action_dict.items():
            if key not in self.action_name_dict.keys():
                continue
            print(key, ": ", end='||')
            for idx in value:
                print(self.action_name_dict[key][idx], end='||')
            print()

    def explain_to_string(self, action_dict):
        exp_list = []
        for key, value in action_dict.items():
            if key not in self.action_name_dict.keys():
                continue
            if exp_list != []:
                exp_list.append('  &&  ')
            exp_list.append(key)
            exp_list.append(": ||")
            exp_sub_list = []
            for idx in value:
                exp_sub_list.append(self.action_name_dict[key][idx])
            exp_sub = '||'.join(exp_sub_list)
            exp_sub += '||'
            exp_list.append(exp_sub)
        # print(exp_list)
        return ''.join(exp_list)


if __name__ == "__main__":
    cola_action_exp = ColaActionExplanation("./config.yaml")
    action_dict = {'hands': [0, 1], 'pose': [1, 2]}
    # cola_action_exp.explain(action_dict)
    action_dict = {'hands': [0, 2], 'pose': [0, 1]}
    print(cola_action_exp.explain_to_string(action_dict))
