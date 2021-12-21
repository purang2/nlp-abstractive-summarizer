# summary termproject
Summarization applications by KNU 2021-2 term project



<!--
### 4 Days Plan

- 21-12-09 목 : TARGET ARTICLE에 대해 Model 성능 확인. Fine-tuning은 이미 되어 있는 모델들임 (data: xsum, arxiv 등)
- 21-12-10 금 : Huggingface application 구동 시키기. Pretrained Language Model 속도적 성능 잡기. Target Dataset 선정 적용 { cord-19, korean }
- 21-12-11 토 : Target Dataset 선정 적용 { cord-19, korean }
- 21-12-12 일 :  마감, 보고서 제출완료~
-->


### @App


**Running on hugging-face Spaces**  
https://huggingface.co/spaces/KNU-Eunchan/Summarizing-app
  
**Programmed by Streamlit**

- API Documents: are here ([https://docs.streamlit.io/library/api-reference/](https://docs.streamlit.io/library/api-reference/)
- Interface : Huggingface Spaces



### Target Models

| Model Name | Pretrained Dataset | Model Size |
| --- | --- | --- |
| Distil-BART-LARGE | CNN News Article |  |
| Pegasus | XSum |  |
| T5-base | - (Original) |  |
|  |  |  |


### Target Article

```python
ARTICLE = """ 
New York (CNN)When Liana Barrientos was 23 years old, 
she got married in Westchester County, New York. 
A year later, she got married again in Westchester County, 
but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. 
Then, Barrientos declared "I do" five more times, sometimes only within two weeks of 
each other.
In 2010, she married once more, this time in the Bronx. 
In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument 
for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, 
according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and 
criminal trespass for allegedly sneaking into the New York subway 
through an emergency exit, said Detective Annette Markowski, a police spokeswoman. 
In total, Barrientos has been married 10 times, with nine of her marriages occurring 
between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. 
She is believed to still be married to four men, and at one time, 
she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, 
who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. 
It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney's Office by Immigration 
and Customs Enforcement and the Department of Homeland Security's
Investigation Division. Seven of the men are from so-called "red-flagged" countries, 
including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan 
after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  
Her next court appearance is scheduled for May 18.
 """
```

### Summarizing Output

**T5-base**

```python
output_text = tokenizer.decode(outputs[0])

print(len(output_text.split(" ")))
>>> 42 #output length (단어 수)

print(output_text)

>>> ''' 
<pad> prosecutors say the marriages were part of an immigration scam. 
if convicted, barrientos faces two criminal 
counts of "offering a false instrument for filing in the first degree" 
she has been married 10 times, 
nine of them between 1999 and 2002.</s>
'''
```

**Pegasus-xsum**

```python
outputs = model.generate(**inputs)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(output_text) 

>>> '''
['A New York woman who has been married 10 times has been charged with marriage fraud.']
'''

print(len(output_text[0].split(" ")))
>>> 16 #output length (단어 수) 

```

**Distil-Bart-cnn**

```python
outputs = model.generate(inputs['input_ids'], num_beams=4, max_length=150, 
early_stopping=True)

output_text = [tokenizer.decode(g, skip_special_tokens=True, 
clean_up_tokenization_spaces=False) for g in outputs]

print(output_text[0])

>>>'''
In total, Liana Barrientos has been married to 10 men since 1999 . 
She faces two counts of "offering a false instrument for filing in the first degree" 
in New York City . She is believed to still be married to four men, and at one time was 
married to eight men .
'''

print(len(output_text[0].split(" ")))
>>> 54 #output length (단어 수)
```
