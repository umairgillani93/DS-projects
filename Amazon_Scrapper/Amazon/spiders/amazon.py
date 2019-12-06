# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
import os

class AmazonSpider(scrapy.Spider):
    name = 'amazon'
    # allowed_domains = ['amazon.com']
    start_urls = [
        'https://www.amazon.com/ask/questions/asin/B0764N2QLF/2/ref=ask_dp_iaw_ql_hza?isAnswered=true#question-Tx1ER766CN8SQSD',
        'https://www.amazon.com/ask/questions/asin/B0764N2QLF/3/ref=ask_ql_psf_ql_hza?isAnswered=true',
    ]

    def parse(self, response):
        items = {}
        titles = response.css('a.a-size-large.a-link-normal::text').extract()
        # rating = response.css('.active').extract()

        questions = response.css('.a-spacing-small .a-declarative::text').extract()
        answers = response.css('noscript+ span::text').extract()
        new_questions = []
        new_answers = []
        new_title = []

        for title in titles:
            title = str(title)
            title = title.replace('\n', '').strip()
            new_title.append(title)

        for question in questions:
            question = str(question)
            question = question.replace('\n', '').strip()
            new_questions.append(question)

        for answer in answers:
            answer = str(answer)
            answer = answer.replace('\n', '').strip()
            new_answers.append(answer)
        print("=" * 20)
        print(len(new_questions))
        print(len(new_answers))
        # print('Title list length: {}'.format(len(title)))
        # title = str(title[0])
        # title = title.replace('\n', '').strip()
        # print('=' * 20)

        items['Product_Title'] = new_title
        # items['rating'] = rating
        items['Questions'] = new_questions
        items['Answers'] = new_answers
        # items['answers']  = answers
        # items['questions'] = questions
        # df = pd.DataFrame({key:pd.Series(value) for key, value in items.items()})
        yield items
        # df.to_csv(os.getcwd() + '\Amazon_Scraped_Data.csv')
        # print(df)
