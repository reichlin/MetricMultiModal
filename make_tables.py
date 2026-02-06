import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


sim_type = "fetch"
rl_type = "sac"
exp_type = "standard_training"  # "standard_training" "training_with_noise"


all_seeds = [0, 1, 2] #[0, 1, 2, 3, 4]
all_envs = [0, 1, 2, 3, 4, 6, 7] if sim_type == "mujoco" else [0, 1, 2, 3]
all_algos = [0, 1, 2, 3, 4, 5, 6]
all_noises = [0, 1, 2, 3, 4, 5, 6]
all_p = [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]

algo_names = {
    0: "LinearComb",
    1: "ConCat",
    2: "Curl",
    3: "MMM",
    4: "GMC",
    5: "AMDF",
    6: "CORAL"
}
if sim_type == "mujoco":
    env_names = {
        0: "Ant-v5",
        1: "HalfCheetah-v5",
        2: "Hopper-v5",
        3: "Humanoid-v5",
        4: "Walker2d-v5",
        5: "Pusher-v5",
        6: "Reacher-v5",
        7: "InvertedPendulum-v5"
    }
elif sim_type == "fetch":
    env_names = {
        0: "FetchReachDense-v",
        1: "FetchPushDense-v",
        2: "FetchPickAndPlaceDense-v",
        3: "FetchSlideDense-v"
    }
all_noises_names = {
    0: "gaussian",
    1: "salt_pepper",
    2: "patches",
    3: "puzzle",
    4: "failure",
    5: "texture",
    6: "hallucination"
}
all_colors_per_model = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
    6: "tab:pink"
}
all_colors_per_noise = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
    6: "tab:pink"
}

all_results = {}

span_rewards_per_env = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: []
}

pickle_name = "./saved_assets/"+sim_type+"/saved_test_results_"+rl_type+"/all_results.pkl"

if not os.path.isfile(pickle_name):
    #
    # for noised_mods in [1, 2]:

    for i_e, env_id in enumerate(all_envs):
        all_results[env_id] = {}
        for i_a, algo in enumerate(all_algos):
            all_results[env_id][algo] = {}
            for i_s, seed in enumerate(all_seeds):
                all_results[env_id][algo][seed] = {}
                name_test_file = "./saved_assets/" + sim_type + "/saved_test_results_" + rl_type + "/"
                name_test_file += 'seed=' + str(seed)
                name_test_file += '_algo=' + str(algo)
                name_test_file += '_env_id=' + str(env_id)
                name_test_file += ".npy"
                if os.path.isfile(name_test_file):
                    r = np.load(name_test_file)
                    for n_i, noise in enumerate(range(r.shape[1])):
                        all_results[env_id][algo][seed][noise] = {}
                        for p_i, p_noise in enumerate(range(r.shape[2])):
                            all_results[env_id][algo][seed][noise][p_noise] = {}
                            for m_i, noised_mods in enumerate(range(r.shape[0])):
                                all_results[env_id][algo][seed][noise][p_noise][noised_mods] = r[m_i, n_i, p_i]
                                span_rewards_per_env[env_id].append(r[m_i, n_i, p_i])
                else:
                    print('seed=' + str(seed) + '_algo=' + str(algo) + '_env_id=' + str(env_id), end=" ")
                    print('DOES NOT EXISTS')

    # for noised_mods in [1, 2]:
    #
    #     for i_e, env_id in enumerate(all_envs):
    #         all_results[env_id] = {}
    #         for i_a, algo in enumerate(all_algos):
    #             all_results[env_id][algo] = {}
    #             for i_n, noise in enumerate(all_noises):
    #                 if noised_mods == 2 and noise == 5:
    #                     continue
    #                 all_results[env_id][algo][noise] = {}
    #                 for i_p, p_noise in enumerate(all_p):
    #                     all_results[env_id][algo][noise][p_noise] = {}
    #                     for i_s, seed in enumerate(all_seeds):
    #                         name_test_file = "./saved_assets/"+sim_type+"/saved_test_results_"+rl_type+"/"
    #                         name_test_file += 'seed=' + str(seed)
    #                         name_test_file += '_algo=' + str(algo)
    #                         name_test_file += '_env_id=' + str(env_id)
    #                         name_test_file += '_noise=' + str(noise)
    #                         name_test_file += '_p=' + str(p_noise)
    #                         name_test_file += '_noised_mods=' + str(noised_mods)
    #                         name_test_file += ".npy"
    #                         if os.path.isfile(name_test_file):
    #                             r = np.load(name_test_file)
    #                             all_results[env_id][algo][noise][p_noise][seed] = r
    #                             span_rewards_per_env[env_id].append(r[0])
    #                         else:
    #                             print('seed=' + str(seed) + '_algo=' + str(algo) + '_env_id=' + str(env_id) + '_noise=' + str(noise) + '_p=' + str(p_noise) + '_mods=' + str(noised_mods), end=" ")
    #                             print('DOES NOT EXISTS')

        # with open(pickle_name, "wb") as f:  # always binary mode
        #     pickle.dump([all_results, span_rewards_per_env], f, protocol=pickle.HIGHEST_PROTOCOL)

else:

    with open(pickle_name, "rb") as f:
        all_results, span_rewards_per_env = pickle.load(f)


# # PER NOISE
# for noised_mods in [1, 2]:
#     for env_id in all_results.keys():
#         for noise in all_noises:
#             if noise == 5 and noised_mods == 2:
#                 continue
#             plt.figure(figsize=(6.4, 4.8))
#             for algo in all_results[env_id].keys():
#                 model_r = []
#                 x_noises = []
#                 for pi, p_noise in enumerate(all_p):
#
#                     # rewards = np.array([x[0] for x in all_results[env_id][algo][noise][p_noise].values()])
#                     rewards = np.array([x[noise][pi][noised_mods-1] for x in all_results[env_id][algo].values()])
#                     rewards = np.sort(rewards)[-3:]
#                     model_r.append(rewards)
#                     x_noises.append(p_noise)
#
#                 model_r = np.stack(model_r, 0)
#                 plt.plot(x_noises, np.mean(model_r, -1), label=algo_names[algo], color=all_colors_per_noise[algo], linewidth=2)
#                 plt.fill_between(x_noises, np.mean(model_r, -1) - np.std(model_r, -1), np.mean(model_r, -1) + np.std(model_r, -1), alpha=0.2, color=all_colors_per_noise[algo])
#                 plt.scatter(x_noises, np.mean(model_r, -1), color=all_colors_per_noise[algo], s=30, zorder=3)
#
#             folder_name = "./figs/"+sim_type+"_" + rl_type + "_" + exp_type
#             if not os.path.exists(folder_name):
#                 os.mkdir(folder_name)
#
#             plt.title(all_noises_names[noise] + " " + env_names[env_id])
#             #plt.ylim(np.min(np.array(span_rewards_per_env[env_id])), np.max(np.array(span_rewards_per_env[env_id])))
#             plt.legend(loc="lower center",
#                        bbox_to_anchor=(0.5, -0.35),
#                        ncol=4,
#                        frameon=False)
#             plt.subplots_adjust(bottom=0.25)
#             plt.tight_layout()
#             plt.savefig(folder_name + "/" + env_names[env_id] + "_" + all_noises_names[noise] + "_n_mods=" + str(noised_mods) + ".pdf")
#             plt.close()
#
# exit()


# PER NOISE
for noised_mods in [1, 2]:
    print("Number of Noised mods: " + str(noised_mods))
    print()
    for env_id in all_results.keys():
        print(env_names[env_id])
        for noise in all_noises:
            if noise == 5 and noised_mods == 2:
                continue
            print(all_noises_names[noise], "")
            for algo in all_results[env_id].keys():
                print(algo, end=" ")
                for pi, p_noise in enumerate(all_p):

                    # rewards = np.array([x[0] for x in all_results[env_id][algo][noise][p_noise].values()])
                    rewards = np.array([x[noise][6-pi][noised_mods-1] for x in all_results[env_id][algo].values()])
                    rewards = np.sort(rewards)[-3:]
                    print(round(np.mean(rewards, -1), 2), "+-", round(np.std(rewards, -1), 2), end=" ")
                print()

exit()


###############################################################################3



all_r = np.zeros((3, 4, 6, 3, 10000))
all_t = np.zeros((3, 4, 6, 3))

folder_name = './saved_rewards_sac_noise/'
for e_i, env in enumerate([0, 1, 2]):
    for m_i, model in enumerate([0, 1, 2, 4]):
        for n_i, noise in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            for s_i, seed in enumerate([0, 1, 2]):
                file_name = 'reload=0_save_model=0_seed=' + str(seed)
                file_name += '_' + algo_names[model]
                file_name += '_z_dim=64_rl_algo=0_env_id=' + str(env)
                file_name += '_noise_level=' + str(noise)
                file_name += '_render=1_modalities=3_no_state=1_.npy'
                if os.path.isfile(folder_name+file_name):
                    r = np.load(folder_name+file_name)
                    all_r[e_i, m_i, n_i, s_i, :r.shape[0]] = r
                    all_t[e_i, m_i, n_i, s_i] = r.shape[0]
                else:
                    print(file_name, ' DOES NOT EXISTS!')

print()

for e_i, env in enumerate([0, 1, 2]):
    min_time_per_env = int(np.min(all_t[e_i]))
    for m_i, model in enumerate([0, 1, 2, 4]):
        for n_i, noise in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            avg_r_per_model_per_noise = np.mean(all_r[e_i, m_i, n_i, :, :min_time_per_env], -2)
            std_r_per_model_per_noise = np.std(all_r[e_i, m_i, n_i, :, :min_time_per_env], -2)
            m = 5
            avg_r_per_model_per_noise_smooth = np.convolve(avg_r_per_model_per_noise, np.ones(m) / m, mode='valid')[:-m]
            std_r_per_model_per_noise_smooth = np.convolve(std_r_per_model_per_noise, np.ones(m) / m, mode='valid')[:-m]
            plt.plot(avg_r_per_model_per_noise_smooth, label='noise='+str(noise))#label=algo_names[model]+'_noise='+str(noise))
            plt.fill_between(range(avg_r_per_model_per_noise_smooth.shape[0]), avg_r_per_model_per_noise_smooth-std_r_per_model_per_noise_smooth, avg_r_per_model_per_noise_smooth+std_r_per_model_per_noise_smooth, alpha=0.2)
        plt.legend(loc="lower center",
                   bbox_to_anchor=(0.5, -0.35),
                   ncol=3,
                   fontsize=8,
                   frameon=False)
        plt.subplots_adjust(bottom=0.25)
        plt.title('preprocessor='+algo_names[model]+' env='+env_names[env])
        file_name = './figs/training_with_noise/'+algo_names[model]+'_'+env_names[env]
        plt.savefig(file_name+'.pdf')
        plt.close()
        # plt.show()

print()











all_seeds = [0, 1, 2, 3, 4]
all_envs = [0, 1, 2, 3, 4, 6, 7]
env_names = {
    0: "Ant-v5",
    1: "HalfCheetah-v5",
    2: "Hopper-v5",
    3: "Humanoid-v5",
    4: "Walker2d-v5",
    5: "Pusher-v5",
    6: "Reacher-v5",
    7: "InvertedPendulum-v5"
}
all_algos = [0, 1, 2, 3, 4, 5, 6]
algo_names = {
    0: "LinearComb",
    1: "ConCat",
    2: "Curl",
    3: "MMM",
    4: "GMC",
    5: "AMDF",
    6: "CORAL"
}
all_noises = [0, 1, 2, 3, 4, 5, 6]
all_noises_names = {
    0: "gaussian",
    1: "salt_pepper",
    2: "patches",
    3: "puzzle",
    4: "failure",
    5: "texture",
    6: "hallucination"
}
all_p = [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]

all_colors_per_model = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
    6: "tab:pink"
}
all_colors_per_noise = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
    6: "tab:pink"
}

all_results = {}

span_rewards_per_env = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: []
}

for i_e, env_id in enumerate(all_envs):
    all_results[env_id] = {}
    for i_a, algo in enumerate(all_algos):
        all_results[env_id][algo] = {}
        for i_n, noise in enumerate(all_noises):
            all_results[env_id][algo][noise] = {}
            for i_p, p_noise in enumerate(all_p):
                all_results[env_id][algo][noise][p_noise] = {}
                for i_s, seed in enumerate(all_seeds):
                    name_test_file = "./saved_test_results/"
                    name_test_file += 'seed=' + str(seed)
                    name_test_file += '_algo=' + str(algo)
                    name_test_file += '_env_id=' + str(env_id)
                    name_test_file += '_noise=' + str(noise)
                    name_test_file += '_p=' + str(p_noise)
                    name_test_file += ".npy"
                    if os.path.isfile(name_test_file):
                        r = np.load(name_test_file)
                        all_results[env_id][algo][noise][p_noise][seed] = r
                        span_rewards_per_env[env_id].append(r[0])
                    else:
                        print('seed=' + str(seed) + '_algo=' + str(algo) + '_env_id=' + str(env_id) + '_noise=' + str(noise) + '_p=' + str(p_noise), end=" ")
                        print('DOES NOT EXISTS')

print()

# PER MODEL
for env_id in all_results.keys():
    for algo in all_results[env_id].keys():
        for noise in all_results[env_id][algo].keys():
            model_r = []
            x_noises = []
            for p_noise in all_results[env_id][algo][noise].keys():

                rewards = np.array([x[0] for x in all_results[env_id][algo][noise][p_noise].values()])
                model_r.append(rewards)
                # mean_r = np.mean(rewards)
                #
                x_noises.append(p_noise)
                # y.append(mean_r)
                #
                # plt.scatter(p_noise, mean_r, color=all_colors_per_model[noise])

            model_r = np.stack(model_r, 0)
            # plt.plot(x_noises, y, label=all_noises_names[noise])
            plt.plot(x_noises, np.mean(model_r, -1), label=all_noises_names[noise], color=all_colors_per_model[noise])
            plt.fill_between(x_noises, np.mean(model_r, -1) - np.std(model_r, -1), np.mean(model_r, -1) + np.std(model_r, -1), alpha=0.2, color=all_colors_per_model[noise])
            plt.scatter(x_noises, np.mean(model_r, -1), color=all_colors_per_model[noise])

        plt.title(algo_names[algo] + " " + env_names[env_id])
        plt.ylim(np.min(np.array(span_rewards_per_env[env_id])), np.max(np.array(span_rewards_per_env[env_id])))
        plt.legend(loc="lower center",
                   bbox_to_anchor=(0.5, -0.35),
                   ncol=4,
                   frameon=False)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig("./figs/per_model/" + env_names[env_id] + "_" + algo_names[algo] + ".pdf")
        plt.close()
        # plt.show()


# PER NOISE
for env_id in all_results.keys():
    for noise in all_noises:
        plt.figure(figsize=(6.4, 4.8))
        for algo in all_results[env_id].keys():
            model_r = []
            x_noises = []
            for p_noise in all_results[env_id][algo][noise].keys():

                rewards = np.array([x[0] for x in all_results[env_id][algo][noise][p_noise].values()])
                rewards = np.sort(rewards)[-3:]
                model_r.append(rewards)
                x_noises.append(p_noise)
                # mean_r = np.mean(rewards)
                #
                # x.append(p_noise)
                # y.append(mean_r)

            #     plt.scatter(p_noise, mean_r, color=all_colors_per_noise[algo])
            # plt.plot(x, y, label=algo_names[algo])
            model_r = np.stack(model_r, 0)
            # plt.plot(x_noises, y, label=all_noises_names[noise])
            plt.plot(x_noises, np.mean(model_r, -1), label=algo_names[algo], color=all_colors_per_noise[algo], linewidth=2)
            plt.fill_between(x_noises, np.mean(model_r, -1) - np.std(model_r, -1), np.mean(model_r, -1) + np.std(model_r, -1), alpha=0.2, color=all_colors_per_noise[algo])
            plt.scatter(x_noises, np.mean(model_r, -1), color=all_colors_per_noise[algo], s=30, zorder=3)

        plt.title(all_noises_names[noise] + " " + env_names[env_id])
        plt.ylim(np.min(np.array(span_rewards_per_env[env_id])), np.max(np.array(span_rewards_per_env[env_id])))
        plt.legend(loc="lower center",
                   bbox_to_anchor=(0.5, -0.35),
                   ncol=4,
                   frameon=False)
        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()
        plt.savefig("./figs/per_noise/" + env_names[env_id] + "_" + all_noises_names[noise] + ".pdf")
        plt.close()
        # plt.show()

print()

































