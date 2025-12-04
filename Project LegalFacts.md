# Project LegalFacts: AI‑Powered Deed Recognition
Our project aimed to solve a critical, emerging problem: **how can we help the Kadaster recognize legal facts in deeds more effectively?**

## The Goal
The primary objective was to build a system that could analyze a deed and automatically identify:  
- Which legal facts are involved (e.g., sale, mortgage, subdivision, seizure, easement, land exchange, purchase agreement)  
- Who is involved and what their role is  
- Which parcels or apartments are affected  

This is essential because the Kadaster distinguishes more than a hundred different types of such events, known as *legal facts* or *document parts*. Correctly recording these changes is the core of the Kadasters work.

## The Model
Since 2019, significant progress has been made in automatically recognizing cadastral designations, persons, and other key data from deeds. However, recognizing legal facts remains far more complex.  

The challenge lies in the fact that notaries and other deed providers have complete freedom in how they phrase and structure documents. There is no fixed format or standard vocabulary, every deed is written differently. This makes automated recognition of legal facts a much harder task than identifying names or parcel numbers.

## Outcome
Recognizing legal facts is the crucial first step in processing a deed into the **BRK (Basic Registration Cadastre)**.
- You must check whether the deed meets registration requirements, and this depends on the legal fact.  
- The data to be extracted varies per legal fact.  
- Processing priority differs—for example, a seizure must be processed faster than a mortgage or transfer.  

If we succeed in automating legal fact recognition, the benefits are clear:  
- Improved quality of the BRK (more reliable and up‑to‑date)  
- Much more efficient processing  
- Far less manual work  


[Go back to Homepage](../README.md)
