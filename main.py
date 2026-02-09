import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()




def main():
    print("Hello from langchain-course!")
information = """ Elon Musk

Article
Talk
Read
View source
View history

Tools
Appearance hide
Text

Small

Standard

Large
Width

Standard

Wide
Color (beta)

Automatic

Light

Dark
Page extended-confirmed-protected
From Wikipedia, the free encyclopedia
For other uses, see Elon Musk (disambiguation).
Elon Musk
FRS

Musk in 2025
Born	Elon Reeve Musk
June 28, 1971 (age 54)
Pretoria, South Africa
Citizenship	
South Africa
Canada
United States (since 2002)
Education	University of Pennsylvania (BA, BS)
Occupations	
CEO and product architect of Tesla
Founder, CEO, and chief engineer of SpaceX
Founder and CEO of xAI
Founder of the Boring Company and X Corp.
Co-founder of Neuralink, OpenAI, Zip2, and X.com (part of PayPal)
President of the Musk Foundation
Political party	Independent
Spouses	
Justine Wilson
​
​(m. 2000; div. 2008)​
Talulah Riley
​
​(m. 2010; div. 2012)​
​
​(m. 2013; div. 2016)​
Children	14,[a] including Vivian Wilson
Parents	
Errol Musk (father)
Maye Musk (mother)
Relatives	Musk family
Awards	Full list
Senior Advisor to the President
In office
January 20, 2025 – May 28, 2025
Serving with Massad Boulos
President	Donald Trump
Elon Musk's voice
Duration: 1 minute and 13 seconds.1:13
Musk on his departure from the Department of Government Efficiency
Recorded May 30, 2025
Signature

	
This article is part of
a series about
Elon Musk
Personal
Companies
Politics
In the arts and media
vte
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

"""

summary_template = """ 
given the information {information} about a person, i want you to create:
1. a short summary
2. two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template = summary_template
)

llm = ChatOpenAI(
    model_name="gpt-5",
    temperature=0
)

# llm = ChatOllama(model="gemma3:270m", temperature=0)


# LCEL USED HERE, the pipe operator | is used to chain the prompt template and the llm
chain = summary_prompt_template | llm
# the resulting chain from this is called a runnable which can be invoked
response = chain.invoke({"information": information})

print(response.content)


if __name__ == "__main__":
    main()
