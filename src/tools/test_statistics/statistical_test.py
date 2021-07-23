import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data_dir = '/Users/narenbao/workspace/second_journal/csv'
csv_file = os.path.join(data_dir, 'subjective_risk_data_collection')

rsc_csv = os.path.join(data_dir, 'pmpc_risk_score.csv')
pmpc_csv = os.path.join(data_dir, 'rsc_risk_score.csv')


def test_subjective_risk_change():
    pmpc_risk_score = pd.read_csv(pmpc_csv)
    #print(pmpc_risk_score.head(5))
    pmpc_risk_score_data = pmpc_risk_score.iloc[:,1:].to_numpy()
    #print(pmpc_risk_score_data)

    rsc_risk_score = pd.read_csv(rsc_csv)
    #print(rsc_risk_score.head(5))
    driver_id = rsc_risk_score.iloc[:,0].to_numpy().astype(float)
    rsc_risk_score_data = rsc_risk_score.iloc[:,1:].to_numpy()
    #print(driver_id)


    fig, ax = plt.subplots(3, 1, figsize=(8,5))

    plt.rcParams["font.size"] = 15
    ax[0].plot(driver_id, rsc_risk_score_data, marker="o")
    ax[0].set_title('Subjective risk score of RSC')
    ax[0].set_ylim([0,6])
    ax[0].grid(True)

    ax[1].plot(driver_id, stats.zscore(rsc_risk_score_data,axis=1), marker="o")
    ax[1].set_title('Normalized subjective risk score of RSC')
    ax[1].grid(True)
    ax[1].set_ylabel('Subjective risk score [1-5]', fontsize=15)


    ax[2].plot(driver_id, np.mean(rsc_risk_score_data,axis=1), 'ks-', markerfacecolor='w', markersize=8)
    ax[2].set_title('Avaraged subjective risk score of RSC')
    ax[2].set_xlabel('Driver ID', fontsize=15)
    ax[2].set_ylim([0,6])
    ax[2].grid(True)


    plt.tight_layout()
    plt.xlabel('Driver ID', fontsize=15)
    plt.grid(True)
    plt.rcParams["font.size"] = 15
    plt.savefig('fig/RSC_risk.png')
    #plt.show()


    plt.figure(figsize=(5,5))
    plt.rcParams["font.size"] = 15
    plt.imshow(rsc_risk_score_data, aspect="auto")
    plt.xticks([])
    plt.xlabel('Scenario ID')
    plt.ylabel('Driver ID')
    plt.colorbar()

    plt.savefig('fig/RSC_risk_heat.png')
    #plt.show()
    
    ################################################3
    fig, ax = plt.subplots(3, 1, figsize=(8,5))

    plt.rcParams["font.size"] = 15
    ax[0].plot(driver_id, pmpc_risk_score_data, marker="o")
    ax[0].set_title('Subjective risk score of PMPC')
    ax[0].set_ylim([0,6])
    ax[0].grid(True)

    ax[1].plot(driver_id, stats.zscore(pmpc_risk_score_data,axis=1), marker="o")
    ax[1].set_title('Normalized subjective risk score of PMPC')
    ax[1].grid(True)
    ax[1].set_ylabel('Subjective risk score [1-5]', fontsize=15)


    ax[2].plot(driver_id, np.mean(pmpc_risk_score_data,axis=1), 'ks-', markerfacecolor='w', markersize=8)
    ax[2].set_title('Avaraged subjective risk score of PMPC')
    ax[2].set_xlabel('Driver ID', fontsize=15)
    ax[2].set_ylim([0,6])
    ax[2].grid(True)

    plt.tight_layout()
    plt.xlabel('Driver ID', fontsize=15)
    plt.grid(True)
    plt.rcParams["font.size"] = 15
    plt.savefig('fig/PMPC_risk.png')
    #plt.show()


    plt.figure(figsize=(5,5))
    plt.rcParams["font.size"] = 15
    plt.imshow(pmpc_risk_score_data, aspect="auto")
    plt.xticks([])
    plt.xlabel('Scenario ID')
    plt.ylabel('Driver ID')
    plt.colorbar()

    plt.savefig('fig/PMPC_risk_heat.png')
    #plt.show()


    plt.figure(figsize=(10,3))
    plt.rcParams["font.size"] = 15
    plt.plot(driver_id, np.mean(pmpc_risk_score_data,axis=1), 'ks-', markerfacecolor='w', markersize=8, label='PMPC')
    plt.plot(driver_id, np.mean(rsc_risk_score_data,axis=1), 'ks-', markerfacecolor='b', markersize=8, label='RSC')
    plt.ylim([0.5,5.5])
    plt.grid(True)
    plt.ylabel('Avaraged $\mathcal{R}_p$')
    plt.savefig('fig/compare_risk.png')
    plt.legend()
    plt.tight_layout()
    plt.xlabel('Driver ID')
    plt.show()

def test_subjective_risk_change2():
    pmpc_risk_score = pd.read_csv(pmpc_csv)
    #print(pmpc_risk_score.head(5))
    pmpc_risk_score_data = pmpc_risk_score.iloc[:,1:].to_numpy()
    #print(pmpc_risk_score_data)

    rsc_risk_score = pd.read_csv(rsc_csv)
    #print(rsc_risk_score.head(5))
    driver_id = rsc_risk_score.iloc[:,0].to_numpy().astype(float)
    rsc_risk_score_data = rsc_risk_score.iloc[:,1:].to_numpy()
    #print(driver_id)

    fig, ax = plt.subplots(2,1, figsize=(12,6))

    RvsD = np.mean(pmpc_risk_score_data,axis=1) - np.mean(rsc_risk_score_data,axis=1)
    print(RvsD)

    print(RvsD)
    t, p = stats.ttest_1samp(RvsD, 0)

    print(t, p)









