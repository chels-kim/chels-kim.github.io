---
layout: posts
title: "COVID-19 Clinical Trials WorldWide: An EDA"
subtitle: 
mathjax: true
toc: true
author: Chelsea Kim
tags:
  - EDA
  - Jupyter Notebook
  - Python
  - Data science

---

**Hello, world!**

Welcome to my first-ever post on my personal website! A quick intro about myself: I'm Chelsea, and I am a data-enthusiast and an aspiring data scientist basedin Toronto, Canada. I graduated from McGill University, where I jointly studied Physiology and Math during my undergrad and Computational Neuroscience for a Master's degree. And for the past couple of months, I have been trying to practice with more data analysis and ML techniques in the hopes that I can become a data scientist one day. In the meantime, I'll be sharing my journey here on my personal website -- I hope you'll join me if you'd like!

In this post, I'll be running through an exploratory data analysis I've done on publicly-available data of clinical trials being conducted worldwide. The data I used here was originally uploaded on [Kaggle](https://www.kaggle.com/parulpandey/covid19-clinical-trials-dataset). Please note that the `markdown`-converted version of this `jupyter notebook` documentation that I'm writing to upload to my website will not contain the full code that was used to conduct my analyses; for the full version, please see [my github repo](https://github.com/chels-kim/kaggle-covid-clintri)!

---

# Contents

1. [Higher-level analysis of data](#1.-Higher-level-analysis-of-data)
2. [Brainstorming questions I'd like to ask](#2.-Brainstorming-questions-I'd-like-to-ask)
3. [Exploratory data analysis](#3.-Exploratory-analysis)  
    i. [Comparison of COVID-related vs -unrelated studies](#3.1.-Comparison-of-COVID-related-vs.-unrelated-studies)  
    ii. [A deeper dive into COVID-related studies](#3.2.-A-deeper-dive-into-covid-related-studies)

---

Before we get started, here's an overview of what I hope to accomplish in each seciton laid out above.

## Plan of Attack

In my higher-level analysis of the dataset, I'll be taking notes on the author's descriptions on Kaggle. I might also check out the [associated notebook](https://www.kaggle.com/parulpandey/eda-on-covid-19-clinical-trials) to get a better sense of the data. Following this general overview of the dataset, I'll be organizing the burning questions and analysis that I'm inspired to do in an organized manner. By the end of this, I'll have a better idea of what steps I will take throughout the EDA.

After this -- you guessed it -- the EDA itself! Everything leading up to this will have been in preparation for this step. This is where most of my more in-depth analyses will take place. Since I have not much experience with some tools and techniques that I'll be using, my technical goals specifically for this EDA are:
- get more comfortable working with the basic packages like Numpy and Pandas.
- get used to graphing using matplotlib library since that's the most common one.
    
Alright, let's get started!


# 1. Higher-level analysis of data

## Some quick notes on the data

Maintained by the NIH, the database at [ClinicalTrials.gov](ClinicalTrials.gov) contains information about all privately and publicly funded clinical studies around the world. The particular dataset to be used consists of clinical trials related to COVID-19 studies specifically. The authors note the following about their dataset:  
 - XML files: each corresponds to one study; filename is formatted as `NCT########.xml`, where the `#`'s indicate unique numerical identifiers of studies.
 - 1 CSV file: not as detailed as above but provides a summary.
    
Let's see what the data look like.


```python
# Importing relevant modules

# libraries
import pandas as pd
import numpy as np
import os
from xml.etree import ElementTree
import scipy.stats as stats
import datetime

# data viz
import matplotlib.pyplot as plt
import seaborn as sns

# Read file data
current_folder = os.path.dirname(os.path.realpath('__file__'))
xml_path = os.path.join(current_folder, 'covid19-clinical-trials-dataset',
                        'COVID-19 CLinical trials studies', 'COVID-19 CLinical trials studies')

list_of_files = os.listdir(xml_path)
print('Total number of clinical trials:', len(list_of_files))
```

    Total number of clinical trials: 5020
    

That's a lot of clinical trials! And, from what I can see, this is not the complete set of trials available on the website. Since we have a lot of files to deal with, it'll probably be best to extract only relevant information into a Pandas dataframe and work with that throughout the analysis.


```python
# file loaded in a hidden cell above.

# print the first three rows of studies dataframe
df_studies.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>brief_title</th>
      <th>conditions</th>
      <th>date_processed</th>
      <th>enrollment</th>
      <th>id</th>
      <th>intervention</th>
      <th>location_countries</th>
      <th>overall_status</th>
      <th>sponsors</th>
      <th>start_date</th>
      <th>study_type</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Isolation and Culture of Immune Cells and Circ...</td>
      <td>SOLID TUMOR, ADULT; HEALTHY DONORS; COVID DONORS</td>
      <td>February 08, 2021</td>
      <td>1000.0</td>
      <td>NCT00571389</td>
      <td>NaN</td>
      <td>UNITED STATES</td>
      <td>Recruiting</td>
      <td>BioCytics, Inc.</td>
      <td>November 2007</td>
      <td>Observational</td>
      <td>A Study to Facilitate Development of an Ex-Viv...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Collection of Human Samples to Study Hairy Cel...</td>
      <td>HAIRY CELL LEUKEMIA (HCL); CHRONIC LYMPHOCYTIC...</td>
      <td>February 08, 2021</td>
      <td>1263.0</td>
      <td>NCT01087333</td>
      <td>NaN</td>
      <td>UNITED STATES</td>
      <td>Recruiting</td>
      <td>National Cancer Institute (NCI)</td>
      <td>March 2, 2010</td>
      <td>Observational</td>
      <td>Collection of Human Samples to Study Hairy Cel...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Ethnic Differences in the Impact of Breast Can...</td>
      <td>BREAST CANCER</td>
      <td>February 08, 2021</td>
      <td>935.0</td>
      <td>NCT01134172</td>
      <td>SURVEY WEB-BASED OR TELEPHONE INTERVIEW; SURVE...</td>
      <td>UNITED STATES</td>
      <td>Active, not recruiting</td>
      <td>Memorial Sloan Kettering Cancer Center</td>
      <td>May 2010</td>
      <td>Observational</td>
      <td>Breast Cancer and the Workforce: Ethnic Differ...</td>
    </tr>
  </tbody>
</table>
</div>



Hold on -- did I read that right? I thought this dataset only contained COVID-19-related studies, yet looking at the `conditions` column tells me that some of the conditions may not entirely be related to COVID-19. Perhaps this means that later on, I'll have to do more sophisticated categorization of studies. Let me take a better look at the conditions and titles of studies.


```python
pd.set_option('display.max_colwidth', None)
df_studies[['conditions','title']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>conditions</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SOLID TUMOR, ADULT; HEALTHY DONORS; COVID DONORS</td>
      <td>A Study to Facilitate Development of an Ex-Vivo Device Platform for Circulating Tumor Cell and Immune Cell Harvesting, Banking, and Apoptosis-Viability Assay</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HAIRY CELL LEUKEMIA (HCL); CHRONIC LYMPHOCYTIC LEUKEMIA (CLL); NON-HODGKINS LYMPHOMA (NHL); CUTANEOUS T CELL LYMPHOMA (CTCL); ADULT T CELL LYMPHOMA (ATL)</td>
      <td>Collection of Human Samples to Study Hairy Cell and Other Leukemias and to Develop Recombinant Immunotoxins for Cancer Treatment</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BREAST CANCER</td>
      <td>Breast Cancer and the Workforce: Ethnic Differences in the Impact of Breast Cancer on Employment Status, Financial Situation, and Quality of Life (BCW)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ANOGENITAL HERPES; COVID; HERPES LABIALIS</td>
      <td>Viral Infections in Healthy and Immunocompromised Hosts</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIV-1 INFECTION</td>
      <td>A Randomized Comparison of Three Regimens of Chemotherapy With Compatible Antiretroviral Therapy for Treatment of Advanced AIDS-KS in Resource-Limited Settings</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CANCER PATIENTS</td>
      <td>Food: A Three-Arm Randomized Controlled Study Examining Food Insecurity Interventions</td>
    </tr>
    <tr>
      <th>6</th>
      <td>COMPLICATIONS; CESAREAN SECTION</td>
      <td>Vaginal Cleansing Before Cesarean Delivery to Reduce Infection: A Randomized Trial</td>
    </tr>
    <tr>
      <th>7</th>
      <td>COMMUNITY ACQUIRED PNEUMONIA</td>
      <td>Effects of Low-dose Corticosteroids on Survival of Severe Community-acquired Pneumonia</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HYPERTROPHIC CARDIOMYOPATHY; LONG QT SYNDROME</td>
      <td>Exercise in Genetic Cardiovascular Conditions (Lifestyle and Exercise in Hypertrophic Cardiomyopathy "LIVE-HCM"/Lifestyle and Exercise in the Long QT Syndrome "LIVE-LQTS"</td>
    </tr>
    <tr>
      <th>9</th>
      <td>UVEITIS</td>
      <td>Uveitis/Intraocular Inflammatory Disease Biobank (iBank)</td>
    </tr>
  </tbody>
</table>
</div>



Indeed, not all studies seem related to COVID-19; in fact, only the first and fourth entries mention the word "COVID" at all in the `conditions` column . So why were the rest of them that do not study COVID as a condition included in this dataset? To find out, I will randomly select one of the 8 seemingly COVID-irrelevant studies above, and print its brief summary.


```python
pd.reset_option('display.max_colwidth')

sus_id = df_studies.loc[df_studies.conditions == 'CANCER PATIENTS', 'id'].iloc[0]
root = ElementTree.parse(os.path.join(xml_path, sus_id + '.xml')).getroot()
summary = root.find('brief_summary').find('textblock').text

print('This is the brief summary of study #' + sus_id + ', titled "' + df_studies.title.iloc[5] + '":')
print(summary)
```

    This is the brief summary of study #NCT01603316, titled "Food: A Three-Arm Randomized Controlled Study Examining Food Insecurity Interventions":
    
          The investigators have found that many patients getting treatment for cancer have trouble

          getting enough to eat, or do not always have enough money for food. When a patient has these

          problems it can lead to difficulties with completing cancer treatment. Across New York City,

          there are many hospitals that offer their patients food pantry services on location. The

          investigators would like to compare how food pantries within the hospital and two other food

          assistance options: monthly food vouchers and weekly grocery deliveries maybe possible

          solutions to this problem. The patient will be randomly assigned to one of the three

          different food program groups, which means everyone has an equal chance in being in any

          group, like a flip of a coin. The investigators hope to learn how to best help patients with

          trouble getting food and to see if this will help with completing cancer treatment.

    

          The original RCT composed of study arms: 1) hospital -based food pantry (control), 2) food

          voucher program plus access to the food pantry, and 3) grocery delivery program plus access

          to the food pantry will remain open to accrual at Ralph Lauren Cancer Center. The other three

          sites of the original RCT, Lincoln Hospital, Queens Cancer Center and Brooklyn Hospital, have

          reached target accrual. Our modified RCT, to be carried out among an expanded cohort of

          cancer patients is composed of study arms : 1) Food Voucher Program (Voucher); 2) Home

          Grocery Delivery Program (Delivery); and 3) Medically-tailored, Hospital-based Food Pantry

          (Pantry).

    

          For this RCT, we will enroll patients across three Bronx hospitals- Jacobi Medical Center,

          St. Barnabas Hospital, and Montefiore Medical Center. For the new study arms, we will enroll

          patients across Lincoln Medical and Mental Health Center, Jacobi Medical Center, St. Barnabas

          Medical Center, and Montefiore Health System. Before conducting the RCT across Lincoln

          Medical and Mental Health Center and the three new sites in the Bronx, we will refine written

          educational materials to be used in the intervention through focus groups.

        
    

Clearly, the focus of this study appears to be nothing related to the medical conditions associated with COVID-19; rather, it is a study investigating food insecurity in cancer patients and its impact on cancer treatment. In fact, a closer look at the original .xml file reveals where COVID-19 was mentioned at all within the study description:


```python
int_description = root.find('intervention').find('description').text
print('Intervention description: \n\n' + int_description)
```

    Intervention description: 
    
    Each survey will take about 45 minutes.Surveys will include questions on medical treatment, health insurance, work-related information, overall health and well being, eating habits, and satisfaction/use of the food program provided. 3 and 6-month follow-up. All participants will be asked to complete the study contact form. Need Assessment surveys will be administered via telephone or in person. The content of the needs assessment questionnaire has been informed by themes generated through IHCD's ongoing research and community outreach and service activities with the cancer patient population at our participating institutions. The survey will ask participants about the impact on their cancer care and their socioeconomic needs of the COVID-19 crisis.
    

In this description of intervention used in the study, COVID-19 is briefly mentioned in the last sentence as one of the questions to be asked to participants in the survey. Considering the previous summary of the study and how the above paragraph was the only instance where COVID-19 was mentioned by the authors of the study, we can conclude that the study itself is *not* concerned with the SARS-CoV-2 virus, COVID-19, or the physiological complications arising from it. Thus, it's hard to say that this study is one of the "COVID-19 clinical trials" as the dataset is titled.

From this preliminary analysis, I realize that it might be useful for me to first sort through the dataset in the beginning of my EDA, distinguishing studies that do investigate the symptoms, immunological/respiratory effects, or the nature of SARS-CoV-2 virus / COVID-19, from those that do not.

Cool, now we can move onto loading the .csv file as well, which should contain summaries of the .xml files.


```python
# Load the CSV file as well
df_studies_CSV = pd.read_csv( os.path.join(current_folder, 'covid19-clinical-trials-dataset',
                                           'COVID clinical trials.csv'))
df_studies_CSV.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>NCT Number</th>
      <th>Title</th>
      <th>Acronym</th>
      <th>Status</th>
      <th>Study Results</th>
      <th>Conditions</th>
      <th>Interventions</th>
      <th>Outcome Measures</th>
      <th>Sponsor/Collaborators</th>
      <th>...</th>
      <th>Other IDs</th>
      <th>Start Date</th>
      <th>Primary Completion Date</th>
      <th>Completion Date</th>
      <th>First Posted</th>
      <th>Results First Posted</th>
      <th>Last Update Posted</th>
      <th>Locations</th>
      <th>Study Documents</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NCT04595136</td>
      <td>Study to Evaluate the Efficacy of COVID19-0001...</td>
      <td>COVID-19</td>
      <td>Not yet recruiting</td>
      <td>No Results Available</td>
      <td>SARS-CoV-2 Infection</td>
      <td>Drug: Drug COVID19-0001-USR|Drug: normal saline</td>
      <td>Change on viral load results from baseline aft...</td>
      <td>United Medical Specialties</td>
      <td>...</td>
      <td>COVID19-0001-USR</td>
      <td>November 2, 2020</td>
      <td>December 15, 2020</td>
      <td>January 29, 2021</td>
      <td>October 20, 2020</td>
      <td>NaN</td>
      <td>October 20, 2020</td>
      <td>Cimedical, Barranquilla, Atlantico, Colombia</td>
      <td>NaN</td>
      <td>https://ClinicalTrials.gov/show/NCT04595136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NCT04395482</td>
      <td>Lung CT Scan Analysis of SARS-CoV2 Induced Lun...</td>
      <td>TAC-COVID19</td>
      <td>Recruiting</td>
      <td>No Results Available</td>
      <td>covid19</td>
      <td>Other: Lung CT scan analysis in COVID-19 patients</td>
      <td>A qualitative analysis of parenchymal lung dam...</td>
      <td>University of Milano Bicocca</td>
      <td>...</td>
      <td>TAC-COVID19</td>
      <td>May 7, 2020</td>
      <td>June 15, 2021</td>
      <td>June 15, 2021</td>
      <td>May 20, 2020</td>
      <td>NaN</td>
      <td>November 9, 2020</td>
      <td>Ospedale Papa Giovanni XXIII, Bergamo, Italy|P...</td>
      <td>NaN</td>
      <td>https://ClinicalTrials.gov/show/NCT04395482</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NCT04416061</td>
      <td>The Role of a Private Hospital in Hong Kong Am...</td>
      <td>COVID-19</td>
      <td>Active, not recruiting</td>
      <td>No Results Available</td>
      <td>COVID</td>
      <td>Diagnostic Test: COVID 19 Diagnostic Test</td>
      <td>Proportion of asymptomatic subjects|Proportion...</td>
      <td>Hong Kong Sanatorium &amp; Hospital</td>
      <td>...</td>
      <td>RC-2020-08</td>
      <td>May 25, 2020</td>
      <td>July 31, 2020</td>
      <td>August 31, 2020</td>
      <td>June 4, 2020</td>
      <td>NaN</td>
      <td>June 4, 2020</td>
      <td>Hong Kong Sanatorium &amp; Hospital, Hong Kong, Ho...</td>
      <td>NaN</td>
      <td>https://ClinicalTrials.gov/show/NCT04416061</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 27 columns</p>
</div>




```python
# What's in the CSV that's not listed in the XML data? Or at least lost in translation to dfs?
print(df_studies.columns)
print(df_studies_CSV.columns)
```

    Index(['Unnamed: 0', 'brief_title', 'conditions', 'date_processed',
           'enrollment', 'id', 'intervention', 'location_countries',
           'overall_status', 'sponsors', 'start_date', 'study_type', 'title'],
          dtype='object')
    Index(['Rank', 'NCT Number', 'Title', 'Acronym', 'Status', 'Study Results',
           'Conditions', 'Interventions', 'Outcome Measures',
           'Sponsor/Collaborators', 'Gender', 'Age', 'Phases', 'Enrollment',
           'Funded Bys', 'Study Type', 'Study Designs', 'Other IDs', 'Start Date',
           'Primary Completion Date', 'Completion Date', 'First Posted',
           'Results First Posted', 'Last Update Posted', 'Locations',
           'Study Documents', 'URL'],
          dtype='object')
    

Because we extracted only a few important categories of information from the .xml files, the `df_studies` dataframe is more condensed than the .csv file's summary in `df_studies_CSV`.

And with that, I can finally move on to the next step in the EDA: brainstorming questions.

# 2. Brainstorming questions I'd like to ask

In the surface-level analysis of the given dataset, I found that it contains information about some studies that are unrelated to COVID-19. For the sake of brevity, I will be labelling studies related and unrelated to COVID-19 as "COVID-related" and "Unrelated" studies, respectively. With that in mind, here are some questions I would like to ask throughout the analysis:

1. Regarding the entire dataset:
    - What are the conditions that are being studied (in COVID-related and unrelated studies)?
    - What status is each study in? Which of those are COVID-related?
        - From the CSV file column description in original dataset post, there is a very small number of studies with results available (1%). What percentage is this of the actual studies that have already been completed?
    - In which countries do studies take place? Which of those are COVID-related?
    - What are the study types used? Do COVID-related studies differ from the rest?
        - What's the distribution of enrollment sizes for study types?
  

2. In COVID-related studies:
    - Specific to **observational studies**:
        - length of each study (from CSV)
        - how is the outcome measured?
    - Specific to **interventional studies**:
        - length of each study (also from CSV)
        - what were some types of interventions used?
        - in drug studies, what were the top 5 most popular ones studied?
    - In the small percentage of studies that actually have results posted, were any of them focused on COVID?
        - If so, can we draw any meaningful conclusions?



# 3. Exploratory analysis

## 3.1. Comparison of COVID-related vs. unrelated studies

### Conditions being studied

First, I want to categorize the conditions being studied; mainly, are they related to COVID-19 or not? If they are not, then could they still be studying related viruses, such as similar respiratory infections? To find out, I will first parse through the `conditions` column of the `df_studies` dataframe. In doing so, I will break them down into three categories: `COVID-related`, `Respiratory`, and `Unrelated` diseases. The reason for having the second category is because there may be studies that do not focus on COVID-19 specifically, but do study related respiratory diseases.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>conditions</th>
      <th>study_topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5020</td>
      <td>5020</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2464</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>COVID</td>
      <td>COVID-related</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1448</td>
      <td>3935</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the data

condition_detailed_counts = df_studies.study_topic.value_counts()
# print(condition_detailed_counts)

values = condition_detailed_counts.to_numpy()
labels = condition_detailed_counts.keys().to_numpy()

def pctvalue(pct, data):
    counts = int(pct/100 * np.sum(data))
    return "{:.1f}%\n({:d} studies)".format(pct, counts)

fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'))

wedges, texts, autotexts = ax.pie(
    values,
    textprops=dict(color='w'),
    autopct=lambda pct: pctvalue(pct, values)
)

# Colors
def data_color(datalength, mapname='plasma'):
    colours_list = plt.get_cmap(mapname)(np.linspace(0.15,0.84,datalength))
    return colours_list

colours_list = data_color(len(wedges))
for idx, wedge in enumerate(wedges):
    wedge = wedge.set_color(colours_list[idx])

ax.legend(
    wedges, labels,
    title = "Type of condition",
    loc = "center",
    bbox_to_anchor=(0,-0.1, 1,-0.1)
)

ax.set_title("What types of conditions were studied? - An initial breakdown")

plt.show()

# TODO: (BONUS) Make the plot interactive using plotly and enable display of all the raw condition names as listed in the datset

```


    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_20_0.png">
    


This pie chart shows that almost 78% of the current clinical studies available in the dataset are concerned with SARS-CoV-2 virus or the COVID-19 disease.
However, the **conditions** being studied may not completely represent all studies that are related to COVID.
For this, let's examine the **study titles** to see if the `Unrelated` and `Respiratory` studies contain any COVID-related keywords in their titles.

    The titles of following studies suggest that they might be related to COVID-19:
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>brief_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>NCT03042143</td>
      <td>Repair of Acute Respiratory Distress Syndrome by Stromal Cell Administration (REALIST): An Open Label Dose Escalation Phase 1 Trial Followed by a Randomized, Double-blind, Placebo-controlled Phase 2 Trial (COVID-19)</td>
      <td>Repair of Acute Respiratory Distress Syndrome by Stromal Cell Administration (REALIST) (COVID-19)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>NCT03226886</td>
      <td>TRACERx Renal (TRAcking Renal Cell Carcinoma Evolution Through Therapy (Rx)) CAPTURE: COVID-19 Antiviral Response in a Pan-tumour Immune Study</td>
      <td>TRACERx Renal CAPTURE Sub-study</td>
    </tr>
    <tr>
      <th>69</th>
      <td>NCT03708718</td>
      <td>A Phase II Randomised Study of Oral Prednisolone in Early Diffuse Cutaneous Systemic Sclerosis (Initially Double-blind, Then Switched to Open-label Because of Covid-19)</td>
      <td>Prednisolone in Early Diffuse Systemic Sclerosis</td>
    </tr>
    <tr>
      <th>75</th>
      <td>NCT03738774</td>
      <td>Addressing Post-Intensive Care Syndrome Among Survivors of COVID (APICS-COVID)</td>
      <td>Addressing Post-Intensive Care Syndrome Among Survivors of COVID (APICS-COVID)</td>
    </tr>
    <tr>
      <th>125</th>
      <td>NCT04148430</td>
      <td>A Phase II Study of IL-1 Receptor Antagonist Anakinra to Prevent Severe Neurotoxicity and Cytokine Release Syndrome in Patients Receiving CD19-Specific Chimeric Antigen Receptor (CAR) T Cells And to Treat Systemic Inflammation Associated With COVID-19</td>
      <td>A Study of Anakinra to Prevent or Treat Severe Side Effects for Patients Receiving CAR-T Cell Therapy</td>
    </tr>
    <tr>
      <th>155</th>
      <td>NCT04255940</td>
      <td>Impact of a Novel Coronavirus (2019-nCoV) Outbreak on Public Anxiety and Cardiovascular Disease Risk in China</td>
      <td>2019-nCoV Outbreak and Cardiovascular Diseases</td>
    </tr>
    <tr>
      <th>156</th>
      <td>NCT04256395</td>
      <td>Registry Study on the Efficacy of a Self-test and Self-alert Applet in Detecting Susceptible Infection of COVID-19 --a Population Based Mobile Internet Survey</td>
      <td>Efficacy of a Self-test and Self-alert Mobile Applet in Detecting Susceptible Infection of COVID-19</td>
    </tr>
  </tbody>
</table>
</div>



As shown here, studies that were not directly concerned with conditions associated with COVID-19 as classified
through `conditions` category were still focused on the secondary impacts of the disease (e.g. "Impact of a Novel Coronavirus (2019-nCoV) Outbreak on Public Anxiety and Cardiovascular Disease Risk in China"). On the other hand, sorting using the `title` column doesn't seem perfect, as it contains some studies whose main focus is irrelevant (e.g. "A Phase II Randomised Study of Oral Prednisolone in Early Diffuse Cutaneous Systemic Sclerosis (Initially Double-blind, Then Switched to Open-label Because of Covid-19)").

Fortunately, it seems that this distinction can be made easily by sorting using the `brief_title` column instead; this makes sense -- in the most concise version of the title, it would be best not to include a minor detail that is irrelevant to the primary focus of the study (such as in the above example where COVID-19 only impacted the type of trial). While this categorization may not be perfect either, for the sake of simplicity I will use this column to sort through the `Unrelated` and `Respiratory` categories. Below are some example titles from doing so.


```python
# Look for the COVID keywords in titles.
brftitles_unrelated = df_studies[df_studies['study_topic'] == 'Unrelated']['brief_title']
brftitles_resp = df_studies[df_studies['study_topic'] == 'Respiratory']['brief_title']

sus_brftitles_id = []

for brftitle in brftitles_unrelated.append(brftitles_resp):
    if any(key in brftitle.lower() for key in covid_keys):
        sus_brftitles_id.append(df_studies.loc[df_studies['brief_title'] == brftitle,'id'].values[0])

ind = df_studies.index[df_studies['id'].isin(sus_brftitles_id)]
        
# Print some titles
pd.set_option('display.max_colwidth', None)

print('The brief titles of following studies suggest that they might be related to COVID-19:\n')
df_studies.loc[ind,['id','title','brief_title']].head(7)

# pd.set_option('display.max_colwidth', 40)
```

    The brief titles of following studies suggest that they might be related to COVID-19:
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>brief_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>NCT03042143</td>
      <td>Repair of Acute Respiratory Distress Syndrome by Stromal Cell Administration (REALIST): An Open Label Dose Escalation Phase 1 Trial Followed by a Randomized, Double-blind, Placebo-controlled Phase 2 Trial (COVID-19)</td>
      <td>Repair of Acute Respiratory Distress Syndrome by Stromal Cell Administration (REALIST) (COVID-19)</td>
    </tr>
    <tr>
      <th>75</th>
      <td>NCT03738774</td>
      <td>Addressing Post-Intensive Care Syndrome Among Survivors of COVID (APICS-COVID)</td>
      <td>Addressing Post-Intensive Care Syndrome Among Survivors of COVID (APICS-COVID)</td>
    </tr>
    <tr>
      <th>100</th>
      <td>NCT03963622</td>
      <td>Careful Ventilation in Acute Respiratory Distress Syndrome</td>
      <td>Careful Ventilation in Acute Respiratory Distress Syndrome (COVID-19)</td>
    </tr>
    <tr>
      <th>155</th>
      <td>NCT04255940</td>
      <td>Impact of a Novel Coronavirus (2019-nCoV) Outbreak on Public Anxiety and Cardiovascular Disease Risk in China</td>
      <td>2019-nCoV Outbreak and Cardiovascular Diseases</td>
    </tr>
    <tr>
      <th>156</th>
      <td>NCT04256395</td>
      <td>Registry Study on the Efficacy of a Self-test and Self-alert Applet in Detecting Susceptible Infection of COVID-19 --a Population Based Mobile Internet Survey</td>
      <td>Efficacy of a Self-test and Self-alert Mobile Applet in Detecting Susceptible Infection of COVID-19</td>
    </tr>
    <tr>
      <th>159</th>
      <td>NCT04260308</td>
      <td>A Survey of Psychological Status of Medical Workers and Residents in the Context of 2019 Novel Coronavirus Pneumonia in Wuhan, China</td>
      <td>A Survey of Psychological Status of Medical Workers and Residents in the Context of 2019 Novel Coronavirus Pneumonia</td>
    </tr>
    <tr>
      <th>167</th>
      <td>NCT04264533</td>
      <td>Vitamin C Infusion for the Treatment of Severe 2019-nCoV Infected Pneumonia: a Prospective Randomized Clinical Trial</td>
      <td>Vitamin C Infusion for the Treatment of Severe 2019-nCoV Infected Pneumonia</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Label these titles as COVID-related
for brftitle in brftitles_unrelated.append(brftitles_resp):
    if any(key in brftitle.lower() for key in covid_keys):
        df_studies.loc[df_studies['brief_title'] == brftitle,'study_topic'] = 'COVID-related'
```

What do the remaining titles look like?

    There are 441 titles that remain unassociated with COVID-19, with samples below:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>brief_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>NCT02517489</td>
      <td>Effects of Low-dose Corticosteroids on Survival of Severe Community-acquired Pneumonia</td>
      <td>Community-Acquired Pneumonia : Evaluation of Corticosteroids</td>
    </tr>
    <tr>
      <th>23</th>
      <td>NCT03102034</td>
      <td>Phase I Placebo-Controlled Study of the Infectivity, Safety and Immunogenicity of a Single Dose of a Recombinant Live-Attenuated Respiratory Syncytial Virus Vaccine, D46/NS2/N/ΔM2-2-HindIII, Lot RSV#011B, Delivered as Nose Drops to RSV-Seronegative Infants 6 to 24 Months of Age</td>
      <td>Evaluating the Infectivity, Safety, and Immunogenicity of a Single Dose of a Recombinant Live-Attenuated Respiratory Syncytial Virus Vaccine (D46/NS2/N/ΔM2-2-HindIII) in RSV-Seronegative Infants 6 to 24 Months of Age</td>
    </tr>
    <tr>
      <th>54</th>
      <td>NCT03465280</td>
      <td>Airway Intervention Registry (AIR) Extension: Recurrent Respiratory Papillomatosis</td>
      <td>Airway Intervention Registry (AIR): Recurrent Respiratory Papillomatosis (RRP)</td>
    </tr>
    <tr>
      <th>57</th>
      <td>NCT03540706</td>
      <td>Impact of the Use of C-reactive Protein in a Micro-method on the Prescription of Antibiotics in General Practitioners Consulting in the Office</td>
      <td>Impact of the Use of CRP on the Prescription of Antibiotics in General Practitioners</td>
    </tr>
    <tr>
      <th>74</th>
      <td>NCT03734237</td>
      <td>A Pragmatic Assessment of Influenza Vaccine Effectiveness in the DoD</td>
      <td>A Pragmatic Assessment of Influenza Vaccine Effectiveness in the DoD</td>
    </tr>
  </tbody>
</table>
</div>



Now, we can visualize the changes in an updated pie chart.


    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_29_0.png">
    


As shown, COVID-related studies, whether directly or indirectly related, make up a little more than 90% of the trials listed on the [ClinicalTrials.gov](ClinicalTrials.gov) website. Strikingly, only about 1% are study unrelated but respiratory diseases, and about 8% appear to be totally unrelated to COVID-19 (about half as many as in the previous classification). The fact that only a very small percentage of respiratory studies remain unrelated to COVID-19 even after further classification (where 126 studies were merged into the `COVID-related` category) demonstrates that, in many cases, respiratory studies in this dataset may not have set out to focus on COVID-19, but were adjusted to account for it in one way or another.

### Study status
What status is each study currently in? We can start by plotting the `overall_status` column as a simple horizontal bar chart.


    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_32_0.png">
    


As shown, almost half of the studies are currently recruiting for participants! It would be interesting to see what proportion of these studies are made up of `COVID-related` ones.

Moreover, while the data in the above chart is sorted in the descending order of counts, the analysis would benefit from grouping these categories for a clearer picture of status distributions across different stages of studies.
Before this grouping is done, I should bring your attention to the fact that the `overall_status` column actually contains information about **recruitment status** as well as **expanded access status**, and they have slightly different implications.

"Recruitment status" is straight forward; it indicates the status that a study is in in terms of participant recruitment, including studies that are not currently recruiting for various possible reasons. On the other hand, "expanded access" refers to a way for patients with serious diseases or conditions who cannot participate in a clinical trial to gain access to a medical product that has not been approved by the U.S. Food and Drug Administration (FDA) (see [ClinicalTrials.gov](ClinicalTrials.gov)). Thus, this is a special type of distinction for studies that are already completed. 


Now, in order to group the statuses, we should first study the various status categories. The following are descriptions  study/recruitment statuses, as defined on the website:

| Status | Description |
|:---:|:---|
| Not yet recruiting | The study has not started recruiting participants. |
| Recruiting | The study is currently recruiting participants. |
| Enrolling by invitation | The study is selecting its participants from a population, or group of people, decided on by the researchers in advance. These studies are not open to everyone who meets the eligibility criteria but only to people in that particular population, who are specifically invited to participate. |
| Active, not recruiting | The study is ongoing, and participants are receiving an intervention or being examined, but potential participants are not currently being recruited or enrolled. |
| Suspended | The study has stopped early but may start again. |
| Terminated | The study has stopped early and will not start again. Participants are no longer being examined or treated. |
| Completed | The study has ended normally, and participants are no longer being examined or treated (that is, the last participant's last visit has occurred). |
| Withdrawn | The study stopped early, before enrolling its first participant. |
| Unknown | A study on ClinicalTrials.gov whose last known status was recruiting; not yet recruiting; or active, not recruiting but that has passed its completion date, and the status has not been last verified within the past 2 years. |

While these distinctions provide a detailed portrayal of a study's recruitment status, as mentioned above it may aid our understanding to group them. Here, I'll divide them into four categories based on the stage of study they are in:
`Not started`, `In preparation / Active`, `Inactive`, or `Finished`.


Secondly, the following are descriptions of expanded access statuses; these could altogether be put under their own `Expanded access` group.

| Status | Description |
|:---:|:---|
| Available | Expanded access is currently available for this investigational treatment, and patients who are not participants in the clinical study may be able to gain access to the drug, biologic, or medical device being studied.
| No longer available | Expanded access was available for this intervention previously but is not currently available and will not be available in the future.
| Temporarily not available | Expanded access is not currently available for this intervention but is expected to be available in the future.
| Approved for marketing | The intervention has been approved by the U.S. Food and Drug Administration for use by the public.


 


```python
# Plot stacked horizontal bar chart
mybarh_stacked_df(grouped_status, ylabel='Study status',
                  title='Status of clinical studies, with a focus on COVID-related studies\n'
                        'and their percentage-makeup in each status category')
```


    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_34_0.png">
    


The above chart gives us a more visually clear overview of which stage of study each trial is in. Most studies are currently `In preparation or Active`, with a majority of them by far in `Recruiting` status and almost 92% of those being COVID-19 related.

The second most common status group is `Not started`, where almost 90% of them are COVID-19 related. With scientists all over the world actively trying to better understand the disease, it is not surprising that almost 900 studies have been planned to take place in the future. 

Interestingly, a close third-most common status group is `Completed` containing more than 700 studies, with almost 92% of those COVID-related. I'm curious to learn more about these past and future COVID-related studies through further analysis; for example, what did we learn from completed studies, and will future studies be based on these results? More to follow in Section 3.2!

Finally, the two least common status groups by far are `Inactive` and `Expanded access` groups. In both of these groups, close to 100% of the studies are COVID-related. Based on these, we can reason about two things: one, it is likely rare in normal, pandemic-free contexts for studies to be in either of these status groups; and two, instances for studies to be in `Expanded access` status in particular is likely extremely rare events -- events where circumstances are dire, such as in a novel viral pandemic where lives are at stake.


### Countries of study
In which countries do studies take place? Let's find out.



```python
countries = list(df_studies['location_countries'].value_counts().index)
country_counts = list(df_studies['location_countries'].value_counts().values)

mybarh_df(countries[:15], country_counts[:15],
           title='Top 15 countries with the most clinical trials logged',
           diffColourIdx=(0,1,2))

```


    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_37_0.png">
    


Of the top 15 countries conducting the most clinical trials in the world, United States, France, and United Kingdom each
placed in top three. The percentage values highlighted for these top three countries are percentage values of the total
number of clinical trials conducted by the top 15 countries shown. As can be seen, these three countries' trials make
up about half of the share of clinical trials being conducted by the top 15 countries.

Which of those are COVID-related ones?



    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_39_0.png">
    



As shown here, in all top 15 countries conducting the most clinical trials in the world according to
[ClinicalTrials.gov](https://clinicaltrials.gov), more than 83% of the studies are COVID-related; this demonstrates the
world-wide effort that is being put into researching and learning about the virus and its impact in a variety of ways.



### Study type
What are the study types used? Do COVID-related studies differ from the rest? Before I investigate, the following are definitions of study types included in the dataset, as defined on the website.

#### Interventional studies (also called clinical trials):
A type of clinical study in which participants are assigned to groups that receive one or more intervention/treatment
(or no intervention) so that researchers can evaluate the effects of the interventions on biomedical or health-related
outcomes. The assignments are determined by the study's protocol. Participants may receive diagnostic, therapeutic, or
other types of interventions.

#### Observational studies (includes patient registries):
A type of clinical study in which participants are identified as belonging to study groups and are assessed for
biomedical or health outcomes. Participants may receive diagnostic, therapeutic, or other types of interventions, but
the investigator does not assign participants to a specific interventions/treatment. A patient registry is a type of
observational study.

#### Expanded access:
As described previously.



```python
mygroupedbar( data=data, data_labels=['Not COVID-related','COVID-related'],
              data_name='Count', group_labels=study_types_new, group_name='Study type',
              title='Study methods used for \nCOVID- and non-COVID-related studies' )
```


    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_43_0.png">
    



Both COVID-related and unrelated studies follow a similar trend; in both cases, `Interventional` studies are the most
popular method, `Observational` studies not including patient registry come second, and `Patient registry` come third. Finally, `Expanded access` studies are the rarest of study types, only seen in a minuscule portion of COVID-related studies and not at all in non-COVID-related ones. As reasoned previously, expanded access might be unlikely to be approved, unless urgently necessary such as in the COVID-19 pandemic.


### Enrollment sizes (per study type and per study topic)
What's the distribution of enrollment sizes across different study methods, and across different study types?
Here, I ignore all enrollment sizes that are not available (such as in the Expanded access cases) or zero.



    
<img src="/assets/img/2021-03-22-covid-clinical-trials_files/output_46_0.png">
    


    For interventional studies, COVID-related and non-COVID-related studies' enrollment sizes did not come from the same distribution. (p-value = 0.515) 
    
    For observational studies, COVID-related and non-COVID-related studies' enrollment sizes did not come from the same distribution. (p-value = 0.940) 
    
    For observational [patient registry] studies, COVID-related and non-COVID-related studies' enrollment sizes came from the same distribution. (p-value = 0.000) 
    
    

That's it for now -- stay tuned for Section 3.2 where I will dive deeper into COVID-related studies in the dataset. Thanks for reading!

