import spacy
from transformers import pipeline
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Step 1: Import necessary libraries

# Step 2: Load the book into a string variable
with open('book.txt', 'r') as file:
    book_contents = file.read()

# Step 3: Preprocess the text using spaCy
nlp = spacy.load('en_core_web_sm')
doc = nlp(book_contents)

# Remove stop words and punctuation
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token.text.lower()) for token in doc if not token.is_stop and token.is_alpha]

# Step 4: Use NLTK to create a document-term matrix
freq_dist = nltk.FreqDist(lemmatized_tokens)
corpus = freq_dist.most_common()

# Step 5: Apply LDA to the document-term matrix
lda_model = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=100)
topics = lda_model.show_topics(num_topics=100, num_words=10)

# Step 6: Use TextRank or LexRank to generate a summary of the text based on the identified topics
parser = PlaintextParser.from_string(book_contents, sumy.tokenizers.Tokenizer('english'))
summarizer = LexRankSummarizer()
summary = summarizer(parser.document, sentences_count=10)

# Step 7: Use BERT or GPT-2 to refine the summary
nlp = pipeline('summarization')
refined_summary = nlp(summary)[0]['summary_text']

# Step 8: Use regular expressions to further refine the summary
refined_summary = re.sub('[^a-zA-Z0-9 \n\.]', '', refined_summary)

# Step 9: Output the final summary to a file or display it in the console
with open('summary.txt', 'w') as file:
    file.write(refined_summary)
print(refined_summary)
