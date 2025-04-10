# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:34:20 2025

@author: Mahdi Haghbayan
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from openai import OpenAI
import textwrap
from hazm import Normalizer, word_tokenize
from docx import Document
import json

# لیست کلمات توقف فارسی
stop_words_farsi = [
    "به", "از", "با", "در", "بر", "است", "هست", "که", "این", "آن",
    "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده",
    "را", "های", "می", "باشد", "آنها", "وی", "او", "ما", "شما", "ایشان",
    "من", "تو", "او", "برای", "هر", "همه", "هیچ", "نه", "نیست", "بود",
    "نبود", "دارد", "ندارد", "بوده", "نبوده", "باید", "نباید", "اگر",
    "اگرچه", "اما", "ولی", "لیکن", "زیرا", "چرا", "چون", "چگونه",
    "چه", "کسی", "کجا", "کی", "آیا", "چراکه", "بنابراین", "درنتیجه",
    "پس", "بلکه", "نیز", "هم", "همچنین", "ضمنا", "علاوه", "علاوه بر",
    "بااینکه", "به‌طوری‌که", "چنانچه", "اگرچه", "درصورتی‌که", "به‌عبارتی",
    "به‌خصوص", "به‌عنوان‌مثال", "به‌طورکلی", "به‌طورمعمول", "به‌طورخاص",
    # سایر کلمات توقف...
]

# تابع استخراج متن از فایل‌های ورد
def extract_text_from_docx(file_path):
    try:
        document = Document(file_path)
        text = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

# تابع پیش‌پردازش متن فارسی
def preprocess_farsi_text(text):
    normalizer = Normalizer()  # نرمال‌سازی حروف فارسی
    text = normalizer.normalize(text)

    # حذف نیم‌فاصله‌ها
    text = re.sub(r"\u200c", " ", text)
    
    # حذف علائم نگارشی اضافی
    text = re.sub(r'[^\w\s]', '', text)
    
    # حذف کلمات توقف
    tokens = word_tokenize(text)
    text = " ".join(word for word in tokens if word not in stop_words_farsi)
    
    return text

# مسیر دسکتاپ (تغییر دهید اگر در سیستم شما متفاوت است)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
""""

# فایل‌های ورودی (مسیر کامل فایل‌ها را مشخص کنید)
files = [
    os.path.join(desktop_path, "DCC-3041901001.docx"),
    os.path.join(desktop_path, "DCC-3031901201.docx"),
    os.path.join(desktop_path, "DCC-3031900902.docx")
]
"""
files = [
    os.path.join(desktop_path, "DCC-3071900403.docx"),
    os.path.join(desktop_path, "DCC-3031900501.docx"),
    os.path.join(desktop_path, "DCC-3031900801.docx")
]



# استخراج و پیش‌پردازش اسناد
documents = []
for file in files:
    extracted_text = extract_text_from_docx(file)
    if extracted_text:
        processed_text = preprocess_farsi_text(extracted_text)  # پیش‌پردازش متن
        documents.append(processed_text)
    else:
        documents.append("")  # افزودن رشته خالی در صورت عدم استخراج

# کوئری مثال
query = "بررسی پرونده خسارت ثالث جانی"

# پیش‌پردازش کوئری
processed_query = preprocess_farsi_text(query)

# تبدیل متن به بردارهای TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words_farsi, max_features=5000)
tfidf_matrix = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([processed_query])

# شباهت کسینوسی بین کوئری و اسناد
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()


# تبدیل شباهت به احتمال با normalize
"""
exp_similarities = np.exp(cosine_similarities - np.max(cosine_similarities))  # برای جلوگیری از سرریز
probabilities = exp_similarities / np.sum(exp_similarities)

# نمایش نتایج
for idx, probability in enumerate(probabilities):
  print(f"Document {os.path.basename(files[idx])}: Probability of relevance = {probability:.2f}")"""
min_cos = cosine_similarities.min()
max_cos = cosine_similarities.max()
normalized_similarities = (cosine_similarities - min_cos) / (max_cos - min_cos)

# تبدیل شباهت‌ها به احتمال‌ها
probabilities = normalized_similarities / normalized_similarities.sum()

# نمایش مقادیر نرمال‌سازی‌شده
for idx, probability in enumerate(probabilities):
    print(f"Document {os.path.basename(files[idx])}: Probability of relevance = {probability:.2f}")
    
threshold=0.6
relevant_docs = [doc for idx, doc in enumerate(documents) if probabilities[idx] >= threshold]

extracted_sections = []

client = OpenAI(
    # This is the default and can be omitted
    #eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMDY5NzczMTcxMzM3ODI5NDYwOSIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMTA1NDEwMiwidXVpZCI6IjQ2NzY2NWNiLTc2ZWYtNGQ3Ny04MGJkLWVmNWVjMDUyMDdkYiIsIm5hbWUiOiJtYWhkaSIsImV4cGlyZXNfYXQiOiIyMDMwLTAzLTI5VDIyOjM1OjAyKzAwMDAifQ.Ekq79eOHQ63tN4c9KUW9bhh6bhyopsDAPcIEW0ZTRdE
    #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
    base_url="https://api.studio.nebius.com/v1/",
    api_key="eyJhE"
)
for doc in relevant_docs:
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                temperature=0.2,
                messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts only relevant sections from documents, without generating new text."},
                {"role": "user", "content": f"{query}\n\nExtract exact relevant sections, including text, tables, and images from this document:\n{textwrap.shorten(doc, width=1000000000)}"},
                 ]
    
            )
            response1 = completion.to_json()
            extracted_sections.append(response1)
        except Exception as e:
            print(f"Error fetching response from ChatGPT API: {e}")
            extracted_sections.append("")
response_dict = json.loads(response1)
message_content = response_dict["choices"][0]["message"]["content"]
print(message_content)
