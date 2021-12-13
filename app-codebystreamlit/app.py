# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:08:50 2021
@author: puran
"""
import streamlit as st
import torch 
#import transformers 
from transformers import pipeline
#from transformers import 
st.header("Header: Abstractive Summarizer Machine!")

st.write("Information")
st.write("__Inputs__: Text your input article!!")
st.write("__Outputs__: Summarizing output text by Google-Pegasus! ")


plms =["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-base" ]


def load_plms(model_name):
    #model_name = "google/pegasus-xsum"
    summarizer = pipeline(task="summarization", model=model_name) 
    
    return summarizer    
    
    
def get_summarizer(summarizer, sequence:str, maximum_tokens:int, minimum_tokens:int):
	output = summarizer(sequence, num_beams=4, max_length=maximum_tokens, min_length=minimum_tokens, do_sample=False)
	return output[0].get('summary_text')



ARTICLE ="""New York (CNN)
When Liana Barrientos was 23 years old, she got married in Westchester County, New York.A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002. All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages. Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
 The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18."""



with st.spinner('Loading Pretrained Models (_please allow for 30 seconds_)...'):
    summarizer_1 = load_plms(plms[0])   
    summarizer_2 = load_plms(plms[1])   
    summarizer_3 = load_plms(plms[2])   
    

with st.form(key="input_area"):
    display_text = ARTICLE + "\n\n" 
    text_input = st.text_area("Input any text you want to summaryize & classify here (keep in mind very long text will take a while to process):", display_text)
    submit_button = st.form_submit_button(label='SUBMIT')



output_text = []

if submit_button:
    with st.spinner('On summarizing !...wait a second please..'):
        output_text.append(get_summarizer(summarizer_1, text_input, 150, 5))
        output_text.append(get_summarizer(summarizer_2, text_input, 150, 5))
        output_text.append(get_summarizer(summarizer_3, text_input, 150, 5))
   
    
    st.markdown("### Outputs are here !:  ")
    
    for i in range(3):
        st.markdown("**"+ plms[i] +"s Output:  **  ")
        st.text(output_text[i])
        st.success(f"{i+1} of 3 are done!")
        
    st.success("Congrats!!! ALL DONE!")
    st.balloons()
    
    balloon_button = st.form_submit_button(label='More Balloon?')

    if balloon_button:    
        st.balloons()
    
    
    
    