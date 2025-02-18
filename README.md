# Trust Me, I’m Wrong: High-Certainty Hallucinations in LLMs

This directory is the code used to generate the results in the paper "Trust Me, I’m Wrong: High-Certainty Hallucinations
in LLMs" by Adi Simhi, Itay Itzhak, Fazl Barez, Gabriel Stanovsky and Yonatan Belinkov.

## Abstract:

Large Language Models (LLMs) often generate outputs that lack grounding in real-world facts, a phenomenon known as
hallucinations.
Prior research has associated hallucinations with model uncertainty, leveraging this relationship for hallucination
detection and mitigation.
In this paper, we challenge the underlying assumption that all hallucinations are associated with uncertainty.
Using knowledge detection and uncertainty measurement methods,
we demonstrate that models can hallucinate with high certainty even when they have the correct knowledge.
We further show that high-certainty hallucinations are consistent across models and datasets, distinctive enough to be
singled out, and challenge existing mitigation methods.
Our findings reveal an overlooked aspect of hallucinations, emphasizing the need to understand their origins and improve
mitigation strategies to enhance LLM safety.


## Run Code:

This code works on linux. 
To run the code, first install the requirements by running the following command:


```bash
conde env create -f environment.yml
```
### Knowledge Detection
To create a knowledge dataset of a given model from TriviaQA/ natural Questions dataset, run the following command:

```bash
python run_all_steps.py --create_knowledge_dataset True --model_name model_name --path_to_datasets datasets/ --dataset_name natural_question/triviaqa
```
### Uncertainty Calculation
To create uncertainty calculations out of knowledge dataset, run the following command:

```bash
python run_all_steps.py --uncertainty_calculation True --model_name model_name --path_to_datasets datasets/ --dataset_name natural_question/triviaqa --method_k_positive alice/child
```

At the end of this step you will have the following files in the datasets folder:
model_name/dataset_name/method_k_positive/factuality_stats.json
model_name/dataset_name/method_k_positive/hallucinations_stats.json

In the factuality_stats.json are all the examples the model generated the correct answer under the given
method_k_positive setting.
In the hallucinations_stats.json are all the examples the model generated a wrong answer under the given
method_k_positive setting.

In each json is a list of dictionaries with the following keys:

* prob: the probability of the first answer token.
* generated: the generated text under the given method_k_positive.
* true_answer: the true answer.
* prob_diff: the difference between the probability of the most likely and second likely next token.
* semantic_entropy: the semantic entropy of the generated text using temperature of 1.
* mean_entropy: the predictive entropy using temperature of 1.
* most_likely_tokens: top 5 most likely tokens.
* temp_generations: are the generations using temperature of 1.
* semantic_entropy_temp_0.5: the semantic entropy of the generated text using temperature of 0.5 instead of 1.
* mean_entropy_temp_0.5: the predictive entropy using temperature of 0.5 instead of 1.
* temp_generations_temp_0.5: are the generations using temperature of 0.5 instead of 1.
* prompt: the prompt used to generate the text.

### Results
To generate the graphs and tables for the results of the uncertainty calculations, run the following command:

```bash
python run_all_steps.py --run_results True  --path_to_datasets datasets/ 
```

Note: The `semantic_entropy` directory is adapted from the original work presented in the paper ["Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"] [Github repository](https://github.com/jlko/semantic_uncertainty). 


