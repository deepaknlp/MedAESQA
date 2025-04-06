import json
import re
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
import spacy
from prettytable import PrettyTable

DEFAULT_SPACY_MODEL = "en_core_web_lg"
nlp = spacy.load(DEFAULT_SPACY_MODEL, exclude=["entity_linker", "entity_ruler", "textcat", "textcat_multilabel", "ner"])


def load_json(path_from_load):
    with open(path_from_load, 'r') as rfile:
        data = json.load(rfile)
    return data


def save_json(data, path_to_save):
    with open(path_to_save, 'w') as wfile:
        json.dump(data, wfile, indent=4)
    print(f"Saved json file as {path_to_save}")


def compute_answer_accuracy(path_to_processed_annotation_file):
    annotated_data = load_json(path_to_processed_annotation_file)

    all_data = defaultdict(lambda: defaultdict(lambda: {"yes_count": 0, 'question_count': 0}))

    classification_labels = defaultdict(lambda: defaultdict(list))

    for ann_item in annotated_data:
        question_id=ann_item['question_id']
        for method_name, item in ann_item['machine_generated_answers'].items():
            y_n_label = item['is_answer_accurate'].lower()
            classification_labels[question_id][method_name].append(y_n_label)

    for question_id, runs in classification_labels.items():
        for run_name, labels in runs.items():

            all_data[question_id][run_name]["question_count"] += 1
            assert len(labels) == 1
            if 'yes' in labels:
                all_data[question_id][run_name]["yes_count"] += 1

    summary_data =defaultdict(lambda: {"yes_count": 0, "question_count": 0})

    for question, runs in all_data.items():
        for run, details in runs.items():
            summary_data[run]["yes_count"] += details["yes_count"]
            summary_data[run]["question_count"] += details["question_count"]
    final_summary_data =defaultdict(lambda: {"Acceptable Answers": 0, "Accuracy": 0})
    for run, details in summary_data.items():
        yes_count = details["yes_count"]
        total_count = details["question_count"]
        percentage = (yes_count / total_count * 100) if total_count > 0 else 0
        final_summary_data[run]["Acceptable Answers"] = yes_count
        final_summary_data[run]["Accuracy"] = round(percentage, 2)

    return final_summary_data


def compute_completeness_scores(path_to_processed_annotation_file):
    def _calculate_average_classification_percentages(classification_data):
        summary_data =defaultdict(lambda: {
            "required_percentage_sum": 0,
            "unnecessary_percentage_sum": 0,
            "inappropriate_percentage_sum": 0,
            "question_count": 0
        })

        for question_id, runs in classification_data.items():
            for run_name, counts in runs.items():
                    total_count = counts["total_count"]
                    if total_count > 0:
                        required_percentage = (counts["required_count"] / total_count) * 100
                        unnecessary_percentage = (counts["unnecessary_count"] / total_count) * 100
                        inappropriate_percentage = (counts["inappropriate_count"] / total_count) * 100

                        summary_data[run_name]["required_percentage_sum"] += required_percentage
                        summary_data[run_name]["unnecessary_percentage_sum"] += unnecessary_percentage
                        summary_data[run_name]["inappropriate_percentage_sum"] += inappropriate_percentage
                        summary_data[run_name]["question_count"] += 1

        final_summary_data = defaultdict(lambda: {
            "Precision": 0,
            "Redundancy": 0,
            "Harmfulness": 0
        })
        for run_name, totals in summary_data.items():
                question_count = totals["question_count"]
                assert question_count == total_dataset_question_count
                if question_count > 0:
                    avg_required = totals["required_percentage_sum"] / question_count
                    avg_unnecessary = totals["unnecessary_percentage_sum"] / question_count
                    avg_inappropriate = totals["inappropriate_percentage_sum"] / question_count
                    final_summary_data[run_name]["Precision"] = round(avg_required, 2)
                    final_summary_data[run_name]["Redundancy"] = round(avg_unnecessary, 2)
                    final_summary_data[run_name]["Harmfulness"] = round(avg_inappropriate, 2)

        return final_summary_data

    annotated_data = load_json(path_to_processed_annotation_file)
    classification_data = defaultdict(lambda: defaultdict(lambda: {
        "required_count": 0,
        "unnecessary_count": 0,
        "inappropriate_count": 0,
        "total_count": 0
    }))

    for ann_item in annotated_data:
        question_id = ann_item['question_id']
        for method_name, ans_item in ann_item['machine_generated_answers'].items():
            for item in ans_item['answer_sentences']:
                if item["answer_sentence_relevance"] == None:
                    classification_data[question_id][method_name]["total_count"] += 1
                    continue
                sentence_classification = item["answer_sentence_relevance"].lower()

                if sentence_classification == "required":
                    classification_data[question_id][method_name]["required_count"] += 1
                elif sentence_classification == "unnecessary":
                    classification_data[question_id][method_name]["unnecessary_count"] += 1
                elif sentence_classification == "inappropriate":
                    classification_data[question_id][method_name]["inappropriate_count"] += 1

                classification_data[question_id][method_name]["total_count"] += 1

    final_summary_data = _calculate_average_classification_percentages(classification_data)
    return final_summary_data

def print_results_table(*datasets):

    if not datasets:
        print("No data to display.")
        return

    merged_data = defaultdict(dict)

    for data in datasets:
        for run, details in data.items():
            for field, value in details.items():
                if field not in merged_data[run]:
                    merged_data[run][field] = value

    fields = ["Run Name"]
    for run, details in merged_data.items():
        fields.extend([field for field in details.keys() if field not in fields])

    table = PrettyTable()
    table.field_names = fields

    for run, details in merged_data.items():
        row = [run] + [details.get(field, "N/A") for field in fields[1:]]
        table.add_row(row)

    print(table)
def compute_answer_recall(path_to_processed_annotation_file, path_to_cluster_file, cluster_type):
    def _get_cluster_id(question_wise_cluster_data, answer_sent_id):
        for cluster_id, cluster in question_wise_cluster_data.items():
            for item in cluster:
                if item['answer_sentence_id'] == answer_sent_id:
                    return cluster_id
        return -1

    def _calculate_question_wise_answer_recall(question_wise_cluster_list, cluster_type, cluster_size=10,
                                               question_size=total_dataset_question_count):
        summary_data = defaultdict(lambda: {
            "answer_completeness_supported_required": 0,
            "answer_completeness_required": 0,
            "answer_completeness_required_and_borderline": 0,
            "question_count": 0
        })

        for question_id, runs in question_wise_cluster_list.items():
            for run_name, details in runs.items():
                    answer_completeness_supported_required = (len(set(
                        details['supported_required_answer_sentences_cluster'])) / cluster_size) * 100
                    answer_completeness_required = (len(set(
                        details['required_answer_sentences_cluster'])) / cluster_size) * 100
                    answer_completeness_required_and_borderline = (len(set(
                        details['required_and_borderline_answer_sentences_cluster'])) / cluster_size) * 100

                    # Aggregate the percentages for each question
                    summary_data[run_name][
                        "answer_completeness_supported_required"] += answer_completeness_supported_required
                    summary_data[run_name]["answer_completeness_required"] += answer_completeness_required
                    summary_data[run_name][
                        "answer_completeness_required_and_borderline"] += answer_completeness_required_and_borderline

                    summary_data[run_name]["question_count"] += 1

        final_summary_data = defaultdict(lambda: defaultdict(lambda: {
            "Recall (with Supported and Required)" + "-" + cluster_type: 0,
            "Recall (with Required)" + "-" + cluster_type: 0,
            "Recall (with Required and Borderline)" + "-" + cluster_type: 0,
        }))
        for run_name, totals in summary_data.items():
                question_count = totals["question_count"]

                avg_supported_required = totals["answer_completeness_supported_required"] / question_size
                avg_required = totals["answer_completeness_required"] / question_size
                avg_required_and_borderline = totals["answer_completeness_required_and_borderline"] / question_size
                final_summary_data[run_name][
                    "Recall (with Supported and Required)" + "-" + cluster_type] = round(avg_supported_required, 2)
                final_summary_data[run_name]["Recall (with Required)" + "-" + cluster_type] = round(
                    avg_required, 2)
                final_summary_data[run_name][
                    "Recall (with Required and Borderline)" + "-" + cluster_type] = round(avg_required_and_borderline,
                                                                                          2)

        return final_summary_data
    annotated_data = load_json(path_to_processed_annotation_file)
    cluster_data = load_json(path_to_cluster_file)

    question_wise_cluster_list =defaultdict(lambda:defaultdict(lambda: {
        "supported_required_answer_sentences_cluster": [],
        "required_answer_sentences_cluster": [],
        "required_and_borderline_answer_sentences_cluster": []

    }))

    for ann_item in annotated_data:
        question_id = ann_item['question_id']
        for method_name, ans_item in ann_item['machine_generated_answers'].items():
            for item in ans_item['answer_sentences']:
                if item["answer_sentence_relevance"] == None or item["answer_sentence_relevance"] == "unnecessary" or item[
                    "answer_sentence_relevance"] == "inappropriate":
                    continue
                if item["answer_sentence_relevance"] == 'required':
                    cluster_id = _get_cluster_id(cluster_data[question_id]['required_answer_sentences_cluster'],
                                                 method_name+'-'+item['answer_sentence_id'])

                    assert cluster_id != -1
                    question_wise_cluster_list[question_id][method_name]['required_answer_sentences_cluster'].append(
                        cluster_id)

                    cluster_id = _get_cluster_id(cluster_data[question_id]['supported_required_answer_sentences_cluster'],
                                                 method_name+'-'+item['answer_sentence_id'])
                    if cluster_id != -1:
                        question_wise_cluster_list[question_id][method_name][
                            'supported_required_answer_sentences_cluster'].append(cluster_id)

                if item["answer_sentence_relevance"] == 'required' or item["answer_sentence_relevance"] == "borderline":
                    cluster_id = _get_cluster_id(cluster_data[question_id]['required_and_borderline_answer_sentences_cluster'],
                                                 method_name+'-'+item['answer_sentence_id'])
                    assert cluster_id != -1
                    question_wise_cluster_list[question_id][method_name][
                        'required_and_borderline_answer_sentences_cluster'].append(cluster_id)

    final_summary_data = _calculate_question_wise_answer_recall(question_wise_cluster_list, cluster_type=cluster_type)
    return final_summary_data



def extract_sentences_with_pmid(text):

    pmid_pattern = r'\[\d+(?:,\s?\d+)*\]'

    pmid_mapping = {}
    placeholder = 'PMID_PLACEHOLDER'
    count = 0

    def replace_pmid(match):
        nonlocal count
        count += 1
        pmid_key = f"{placeholder}_{count}"
        pmid_mapping[pmid_key] = match.group()  # Store original PMID
        return pmid_key  # Replace with placeholder


    text= text.strip()
    modified_text = re.sub(pmid_pattern, replace_pmid, text)
    doc = nlp(modified_text)

    extracted_data = []
    for sent in doc.sents:
        original_sentence = sent.text.strip()  # Store the original sentence
        sentence = original_sentence  # Get the sentence text
        pmids = []

        sorted_pmid_keys = sorted(pmid_mapping.keys(), key=lambda x: int(x.split('_')[-1]), reverse=True)

        # Restore PMIDs by replacing the placeholders with original PMIDs
        for pmid_key in sorted_pmid_keys:
            pmid_value = pmid_mapping[pmid_key]
            if pmid_key in sentence:
                if sentence.startswith(pmid_key):
                    sentence = re.sub(r'\b' + re.escape(pmid_key) + r'\b', '', sentence)
                    original_sentence = re.sub(r'\b' + re.escape(pmid_key) + r'\b', pmid_value, original_sentence)
                    pmids.extend([int(x.strip()) for x in pmid_value.strip('[]').split(',')])  # Collect the original PMID

                else:
                    sentence = re.sub(r'\b' + re.escape(pmid_key) + r'\b', '', sentence)
                    original_sentence = re.sub(r'\b' + re.escape(pmid_key) + r'\b', pmid_value, original_sentence)

                    pmids.extend([int(x.strip()) for x in pmid_value.strip('[]').split(',')])  # Collect the original PMID




        if not pmids:
            pmids = None


        if sentence.strip()=='' and pmids is None:
            continue
        if sentence.strip()=='.' and pmids is None:
            continue
        extracted_data.append({'sentence': sentence.strip(),
                               'pmids': pmids,
                               'original_sentence': original_sentence})

    return extracted_data


def compute_citation_quality(path_to_processed_annotation_file):

    def _calculate_question_wise_citation_quality(question_wise_citation_list, question_size=total_dataset_question_count):
        summary_data = defaultdict(lambda: {
            "citation_coverage": 0,
            "citation_sr_with_valid_generated_citations": 0,
            "citation_cr_with_valid_generated_citations": 0,
            "citation_sr_with_total_generated_citations": 0,
            "citation_cr_with_total_generated_citations": 0,
            "question_count": 0
        })

        for question_id, runs in question_wise_citation_list.items():
                for run_name, details in runs.items():
                    citation_coverage = details['answer_sentences_with_one_or_more_support_citation'] / details[
                        'number_of_generated_answer_sentences'] * 100
                    assert details['number_of_total_valid_citation'] <= details['number_of_total_generated_citation']
                    if details['number_of_total_valid_citation'] > 0:
                        citation_sr_with_valid_generated_citations = details['number_of_support_citation'] / details[
                            'number_of_total_valid_citation'] * 100
                        citation_cr_with_valid_generated_citations = details['number_of_contradict_citation'] / details[
                            'number_of_total_valid_citation'] * 100

                    else:
                        citation_sr_with_valid_generated_citations = 0
                        citation_cr_with_valid_generated_citations = 0

                    if details['number_of_total_generated_citation'] > 0:
                        citation_sr_with_total_generated_citations = details['number_of_support_citation'] / details[
                            'number_of_total_generated_citation'] * 100
                        citation_cr_with_total_generated_citations = details['number_of_contradict_citation'] / details[
                            'number_of_total_generated_citation'] * 100
                    else:
                        citation_sr_with_total_generated_citations = 0
                        citation_cr_with_total_generated_citations = 0

                    summary_data[run_name]["citation_coverage"] += citation_coverage
                    summary_data[run_name][
                        "citation_sr_with_valid_generated_citations"] += citation_sr_with_valid_generated_citations
                    summary_data[run_name][
                        "citation_cr_with_valid_generated_citations"] += citation_cr_with_valid_generated_citations
                    summary_data[run_name][
                        "citation_sr_with_total_generated_citations"] += citation_sr_with_total_generated_citations
                    summary_data[run_name][
                        "citation_cr_with_total_generated_citations"] += citation_cr_with_total_generated_citations

                    summary_data[run_name]["question_count"] += 1


        final_summary_data = defaultdict(lambda: {
            "Citation Coverage": 0,
            "Citation Support Rate": 0,
            "Citation Contradict Rate": 0
        })
        for run_name, totals in summary_data.items():
                question_count = totals["question_count"]
                assert question_count == total_dataset_question_count
                avg_cc = totals["citation_coverage"] / question_size
                avg_csr_total = totals["citation_sr_with_total_generated_citations"] / question_size
                avg_ccr_total = totals["citation_cr_with_total_generated_citations"] / question_size

                final_summary_data[run_name]["Citation Coverage"] = round(avg_cc, 2)

                final_summary_data[run_name]["Citation Support Rate"] = round(avg_csr_total, 2)
                final_summary_data[run_name]["Citation Contradict Rate"] = round(avg_ccr_total, 2)

        return final_summary_data

    annotated_data = load_json(path_to_processed_annotation_file)

    question_wise_citation_list =defaultdict(lambda: defaultdict(lambda: {
        "answer_sentences_with_one_or_more_support_citation": 0,
        "number_of_support_citation": 0,
        "number_of_contradict_citation": 0,
        'number_of_total_valid_citation': 0,
        'number_of_total_generated_citation': 0,
        'number_of_generated_answer_sentences': 0,

    }))

    for ann_item in annotated_data:
        question_id = ann_item['question_id']
        for method_name, ans_item in ann_item['machine_generated_answers'].items():
            for item in ans_item['answer_sentences']:
                generated_citations = []
                if item['citation_assessment'] is not None:
                    for citation_item in item['citation_assessment']:
                            generated_citations.append(int(citation_item['cited_pmid']))

                question_wise_citation_list[question_id][method_name]['number_of_total_generated_citation'] += len(set(generated_citations))


                if item['citation_assessment'] == None:
                    question_wise_citation_list[question_id][method_name]['number_of_generated_answer_sentences'] += 1

                elif type(item['citation_assessment']) == list:
                    question_wise_citation_list[question_id][method_name]['number_of_generated_answer_sentences'] += 1

                    support_pmids = []
                    contradict_pmids = []
                    total_citation_cited_pmids = []
                    for entity in item["citation_assessment"]:
                        total_citation_cited_pmids.append(entity['cited_pmid'])
                        if entity['evidence_relation'] == 'supporting':
                            support_pmids.append(entity['cited_pmid'])
                        elif entity['evidence_relation'] == 'contradicting':
                            contradict_pmids.append(entity['cited_pmid'])
                    support_count = len(set(support_pmids))
                    contradict_count = len(set(contradict_pmids))

                    if support_count > 0:
                        question_wise_citation_list[question_id][method_name]['answer_sentences_with_one_or_more_support_citation'] += 1
                    question_wise_citation_list[question_id][method_name][
                        'number_of_support_citation'] += support_count
                    question_wise_citation_list[question_id][method_name][
                        'number_of_contradict_citation'] += contradict_count
                    question_wise_citation_list[question_id][method_name][
                        'number_of_total_valid_citation'] += len(set(total_citation_cited_pmids))
                else:
                    print("Error in citation entities!}")
                    exit(-1)

    final_summary_data = _calculate_question_wise_citation_quality(question_wise_citation_list)
    return final_summary_data


def compute_document_relevance(path_to_processed_annotation_file):


    def _calculate_question_wise_citation_quality(question_wise_citation_list, question2relevant_citation_list,
                                                  question_size=total_dataset_question_count):
        summary_data = defaultdict(lambda: {
            "recall": 0,
            "precision_valid_citation": 0,
            "precision_total_citation": 0,
            "question_count": 0
        })

        # Calculate percentages per question and aggregate
        for question_id, runs in question_wise_citation_list.items():
            all_relevant_document = len(set(question2relevant_citation_list[question_id]))
            for run_name, details in runs.items():
                assert len(set(details['valid_document_provided'])) <= len(set(details['total_document_provided']))
                recall = len(set(details['relevant_retrieved_document'])) / all_relevant_document * 100
                if len(set(details['valid_document_provided'])) > 0:
                    precision_valid_citation = len(set(details['relevant_retrieved_document'])) / len(
                        set(details['valid_document_provided'])) * 100

                else:
                    precision_valid_citation = 0

                if len(set(details['total_document_provided'])) > 0:
                    precision_total_citation = len(set(details['relevant_retrieved_document'])) / len(set(details[
                                                                                                              'total_document_provided'])) * 100

                else:
                    precision_total_citation = 0

                summary_data[run_name]["recall"] += recall
                summary_data[run_name]["precision_valid_citation"] += precision_valid_citation
                summary_data[run_name]["precision_total_citation"] += precision_total_citation

                summary_data[run_name]["question_count"] += 1


        final_summary_data = defaultdict(lambda: {
            "Recall": 0,
            "Precision": 0
        })
        for run_name, totals in summary_data.items():
                question_count = totals["question_count"]
                assert question_count == total_dataset_question_count
                avg_recall = totals["recall"] / question_size
                avg_prec_total = totals["precision_total_citation"] / question_size

                final_summary_data[run_name]["Recall"] = round(avg_recall, 2)
                final_summary_data[run_name]["Precision"] = round(avg_prec_total, 2)

        return final_summary_data

    annotated_data = load_json(path_to_processed_annotation_file)

    question_wise_citation_list = defaultdict(lambda: defaultdict(lambda: {
        "relevant_retrieved_document": [],
        'valid_document_provided': [],
        'total_document_provided': []

    }))

    question2relevant_citation_list = defaultdict(lambda: [])

    # Loop through each item in the data
    for ann_item in tqdm(annotated_data):
        question_id = ann_item['question_id']
        for method_name, ans_item in ann_item['machine_generated_answers'].items():
            for item in ans_item['answer_sentences']:
                generated_citations = []
                if item['citation_assessment'] is not None:
                    for citation_item in item['citation_assessment']:
                        generated_citations.append(int(citation_item['cited_pmid']))

                question_wise_citation_list[question_id][method_name]['total_document_provided'].extend(
                    generated_citations)

                if item["citation_assessment"] == None:
                    continue
                elif type(item["citation_assessment"]) == list:

                    total_citation_cited_pmids = []
                    total_citation_relevant_pmids = []

                    for entity in item["citation_assessment"]:

                        total_citation_cited_pmids.append(entity['cited_pmid'])
                        if entity['evidence_relation'] == 'supporting' or entity[
                            'evidence_relation'] == 'contradicting' or entity['evidence_relation'] == 'neutral':
                            total_citation_relevant_pmids.append(entity['cited_pmid'])

                    question_wise_citation_list[question_id][method_name][
                        'relevant_retrieved_document'].extend(total_citation_relevant_pmids)

                    question_wise_citation_list[question_id][method_name][
                        'valid_document_provided'].extend(total_citation_cited_pmids)

                    question2relevant_citation_list[question_id].extend(total_citation_relevant_pmids)
                else:
                    print("Error in citation entities!}")
                    exit(-1)

    final_summary_data = _calculate_question_wise_citation_quality(question_wise_citation_list,
                                                                   question2relevant_citation_list)
    return final_summary_data

def main(path_to_processed_annotation_file, path_to_ST_cluster_file, path_to_SimCSE_cluster_file):
    result_accuracy = compute_answer_accuracy(
        path_to_processed_annotation_file=path_to_processed_annotation_file)
    print(f"result_accuracy: ")
    print_results_table(result_accuracy)

    result_rates = compute_completeness_scores(
        path_to_processed_annotation_file=path_to_processed_annotation_file)
    print(f"result_rates: ")
    print_results_table(result_rates)
    result_recall_ST = compute_answer_recall(
        path_to_processed_annotation_file=path_to_processed_annotation_file,
        path_to_cluster_file=path_to_ST_cluster_file,
        cluster_type='ST')
    print(f"result_recall_ST: ")
    print_results_table(result_recall_ST)
    result_recall_SimCSE = compute_answer_recall(
        path_to_processed_annotation_file=path_to_processed_annotation_file,
        path_to_cluster_file=path_to_SimCSE_cluster_file,
        cluster_type='SimCSE')

    print(f"result_recall_SimCSE: ")
    print_results_table(result_recall_SimCSE)

    result_citation_quality = compute_citation_quality(
        path_to_processed_annotation_file=path_to_processed_annotation_file)
    print(f"result_citation_quality: ")
    print_results_table(result_citation_quality)

    result_document_relevance = compute_document_relevance(
        path_to_processed_annotation_file=path_to_processed_annotation_file)
    print(f"result_document_relevance: ")
    print_results_table(result_document_relevance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process three string paths.")
    parser.add_argument("--path_to_processed_annotation_file", type=str, default='../data/medaesqa_v1.json')
    parser.add_argument("--path_to_ST_cluster_file", type=str, default='../data/question_to_clustered_answer_sentences_sentence_transformers_kmeans_transformed.json')
    parser.add_argument("--path_to_SimCSE_cluster_file", type=str, default="../data/question_to_clustered_answer_sentences_sim_cse_kmeans_transformed.json'")

    args = parser.parse_args()

    if 'biogen' in args.path_to_processed_annotation_file:
        total_dataset_question_count=65
    elif 'medaesqa' in args.path_to_processed_annotation_file:
        total_dataset_question_count=40
    else:
        print('Dataset name should be either biogen or medaesqa.')
        exit(-1)

    main(args.path_to_processed_annotation_file, args.path_to_ST_cluster_file, args.path_to_SimCSE_cluster_file)

