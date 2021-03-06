import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
nlp = spacy.load('en_core_web_sm')
# Reference: https://discuss.huggingface.co/t/summarization-on-long-documents/920/7
def create_nest_sentences(document:str, token_max_length = 1024):
  nested = []
  sent = []
  length = 0
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
  tokens = nlp(document)
  for sentence in tokens.sents:
    tokens_in_sentence = tokenizer(str(sentence), truncation=False, padding=False)[0] # hugging face transformer tokenizer
    length += len(tokens_in_sentence)
    if length < token_max_length:
      sent.append(sentence)
    else:
      nested.append(sent)
      sent = []
      length = 0
  if sent:
    nested.append(sent)
  return nested
# Reference: https://huggingface.co/facebook/bart-large-mnli
def load_summary_model():
    model_name = "facebook/bart-large-mnli"
    summarizer = pipeline(task='summarization', model=model_name)
    return summarizer
# def load_summary_model():
#     model_name = "facebook/bart-large-mnli"
#     tokenizer = BartTokenizer.from_pretrained(model_name)
#     model = BartForConditionalGeneration.from_pretrained(model_name)
#     summarizer = pipeline(task='summarization', model=model, tokenizer=tokenizer, framework='pt')
#     return summarizer
def summarizer_gen(summarizer, sequence:str, maximum_tokens:int, minimum_tokens:int):
	output = summarizer(sequence, num_beams=4, max_length=maximum_tokens, min_length=minimum_tokens, do_sample=False)
	return output[0].get('summary_text')
# # Reference: https://www.datatrigger.org/post/nlp_hugging_face/
# # Custom summarization pipeline (to handle long articles)
# def summarize(text, minimum_length_of_summary = 100):
#     # Tokenize and truncate
#     inputs = tokenizer_bart([text], truncation=True, max_length=1024, return_tensors='pt').to('cuda')
#     # Generate summary 
#     summary_ids = model_bart.generate(inputs['input_ids'], num_beams=4, min_length = minimum_length_of_summary, max_length=400, early_stopping=True)
#     # Untokenize
#     return([tokenizer_bart.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0])
# Reference: https://huggingface.co/spaces/team-zero-shot-nli/zero-shot-nli/blob/main/utils.py
def load_model():
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer, framework='pt')
    return classifier
def classifier_zero(classifier, sequence:str, labels:list, multi_class:bool):
    outputs = classifier(sequence, labels, multi_label=multi_class)
    return outputs['labels'], outputs['scores']
