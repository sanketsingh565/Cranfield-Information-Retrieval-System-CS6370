import scipy.stats as stats 
import json


# Load the JSON data from the file into a dictionary
with open("vsm_metrics.json", 'r') as json_file:
    baseline_metrics = json.load(json_file)

with open("lsa_metrics.json", 'r') as json_file:
    lsa_metrics = json.load(json_file)

with open("doc2vec_metrics.json", 'r') as json_file:
    doc2vec_metrics = json.load(json_file)

with open("word2vec_metrics.json", 'r') as json_file:
    word2vec_metrics = json.load(json_file)

with open("bm25_metrics.json", 'r') as json_file:
    bm25_metrics = json.load(json_file)

#Paired-t between baseline and other methods
paired_t_metrics = {} #Dictionary of dictionary
# Performing the paired sample t-test 

paired_t_metrics["precision_at_10"] = {} 
paired_t_metrics["precision_at_10"]["lsa"] = stats.ttest_rel(baseline_metrics["precisions_at_10"], lsa_metrics["precisions_at_10"]) 
paired_t_metrics["precision_at_10"]["doc2vec"] = stats.ttest_rel(baseline_metrics["precisions_at_10"], doc2vec_metrics["precisions_at_10"]) 
paired_t_metrics["precision_at_10"]["word2vec"] = stats.ttest_rel(baseline_metrics["precisions_at_10"], word2vec_metrics["precisions_at_10"]) 
paired_t_metrics["precision_at_10"]["bm25"] = stats.ttest_rel(baseline_metrics["precisions_at_10"], bm25_metrics["precisions_at_10"]) 


paired_t_metrics["recalls_at_10"] = {} 
paired_t_metrics["recalls_at_10"]["lsa"] = stats.ttest_rel(baseline_metrics["recalls_at_10"], lsa_metrics["recalls_at_10"]) 
paired_t_metrics["recalls_at_10"]["doc2vec"] = stats.ttest_rel(baseline_metrics["recalls_at_10"], doc2vec_metrics["recalls_at_10"]) 
paired_t_metrics["recalls_at_10"]["word2vec"] = stats.ttest_rel(baseline_metrics["recalls_at_10"], word2vec_metrics["recalls_at_10"]) 
paired_t_metrics["recalls_at_10"]["bm25"] = stats.ttest_rel(baseline_metrics["recalls_at_10"], bm25_metrics["recalls_at_10"]) 


paired_t_metrics["fscores_at_10"] = {} 
paired_t_metrics["fscores_at_10"]["lsa"] = stats.ttest_rel(baseline_metrics["fscores_at_10"], lsa_metrics["fscores_at_10"]) 
paired_t_metrics["fscores_at_10"]["doc2vec"] = stats.ttest_rel(baseline_metrics["fscores_at_10"], doc2vec_metrics["fscores_at_10"]) 
paired_t_metrics["fscores_at_10"]["word2vec"] = stats.ttest_rel(baseline_metrics["fscores_at_10"], word2vec_metrics["fscores_at_10"]) 
paired_t_metrics["fscores_at_10"]["bm25"] = stats.ttest_rel(baseline_metrics["fscores_at_10"], bm25_metrics["fscores_at_10"]) 


paired_t_metrics["MAPs_at_10"] = {} 
paired_t_metrics["MAPs_at_10"]["lsa"] = stats.ttest_rel(baseline_metrics["MAPs_at_10"], lsa_metrics["MAPs_at_10"]) 
paired_t_metrics["MAPs_at_10"]["doc2vec"] = stats.ttest_rel(baseline_metrics["MAPs_at_10"], doc2vec_metrics["MAPs_at_10"]) 
paired_t_metrics["MAPs_at_10"]["word2vec"] = stats.ttest_rel(baseline_metrics["MAPs_at_10"], word2vec_metrics["MAPs_at_10"]) 
paired_t_metrics["MAPs_at_10"]["bm25"] = stats.ttest_rel(baseline_metrics["MAPs_at_10"], bm25_metrics["MAPs_at_10"]) 


paired_t_metrics["nDCGs_at_10"] = {} 
paired_t_metrics["nDCGs_at_10"]["lsa"] = stats.ttest_rel(baseline_metrics["nDCGs_at_10"], lsa_metrics["nDCGs_at_10"]) 
paired_t_metrics["nDCGs_at_10"]["doc2vec"] = stats.ttest_rel(baseline_metrics["nDCGs_at_10"], doc2vec_metrics["nDCGs_at_10"]) 
paired_t_metrics["nDCGs_at_10"]["word2vec"] = stats.ttest_rel(baseline_metrics["nDCGs_at_10"], word2vec_metrics["nDCGs_at_10"]) 
paired_t_metrics["nDCGs_at_10"]["bm25"] = stats.ttest_rel(baseline_metrics["nDCGs_at_10"], bm25_metrics["nDCGs_at_10"]) 

file_path = "paired_t_results.json"
# Save the dictionary to a JSON file
# t-value, p-value pair
with open(file_path, 'w') as json_file:
    json.dump(paired_t_metrics, json_file)  
