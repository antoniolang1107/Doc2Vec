from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, svm

def load_data(fname):
	document_data = []
	label_data = []
	with open(fname, 'r') as file:
		for record in file:
			split_record = record.split(',', 5)
			label_data.append(split_record[0])
			document_data.append(split_record[5])
	return document_data, label_data

def clean_data(documents, labels) -> (list, list):
	"""Cleans the labels and documents

	:param documents: list of strings
	:param lables: list of 0 or 4 labels
	:return documents, cleaned_labels
	"""
	punctuation = r"!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~"
	cleaned_labels = [1 if label == '4' else 0 for label in labels]
	for index, document in enumerate(documents):
		cleaned_document = []
		for word in document.split(' '):
			if '@' in word or 'http' in word:
				continue
			parsed_word = word.replace('-', ' ')
			parsed_word = ''.join([char.lower() for char in word if char not in punctuation])
			cleaned_document.append(parsed_word)
		documents[index] = ' '.join(cleaned_document)
	return documents, cleaned_labels

def train_doc2vec(cleaned_documents, labels):
	"""Trains and saves a doc2vec model
	
	:param cleaned_documents: list of preprocessed documents
	:return model: trained doc2vec model

	tokenize cleaned dataset using nltk
	train doc2vec from gensim
	save model to disk
	return model
	"""

	tokenized_documents = [word_tokenize(tweet) for tweet in cleaned_documents]
	labled_docs = []
	for index, doc in enumerate(tokenized_documents):
		labled_docs.append(TaggedDocument(words=doc, tags=[labels[index]]))

	fname = "doc2vec"
	model = Doc2Vec(labled_docs)
	model.save(fname)
	return Doc2Vec.load(fname)

def tokenize_data(cleaned_documents, d2v_model) -> list:
	"""Tokenizes the sample data

	:param cleaned_documents: list of preprocessed documents
	:param d2v_model: trained doc2vec model
	:return vectorized_docs: vectorized list of documents
	"""
	return [d2v_model.infer_vector(doc.split(' '), epochs=100) for doc in cleaned_documents]

def train(X_train, y_train) -> dict:
	"""Trains multiple models from scikit-learn
	
	:param X_train: list of training samples
	:param y_train: list of labels
	:return models: {"model": model_object, etc}
	"""
	svm_model = svm.SVC().fit(X_train, y_train)
	decision_tree = tree.DecisionTreeClassifier().fit(X_train, y_train)
	logistic_regression = linear_model.LogisticRegression().fit(X_train, y_train)
	return {"svm" : svm_model,
			"decision_tree" : decision_tree,
			"logistic_regression" : logistic_regression}

def test(trained_models_dict, X_test, y_test):
	pass


if __name__ == "__main__":
	documents, labels = load_data("twitter_sentiment_mini.csv")
	# doc_copy = documents.copy()
	cleaned_docs, cleaned_labels = clean_data(documents, labels)
	# print("--DOCUMENTS--")
	# for i in range(5):
	# 	print(f"Original: {doc_copy[i]}")
	# 	print(f"Cleaned: {cleaned_docs[i]}")
	# print("--LABELS--")
	# print(labels[:5])
	# print(cleaned_labels[:5])
	model = train_doc2vec(cleaned_docs, cleaned_labels)
	vec_docs = tokenize_data(cleaned_docs, model)
	X_train, X_test, y_train, y_test = train_test_split(vec_docs, cleaned_labels, test_size=0.33, random_state=1, stratify=cleaned_labels) 
		# changed strafity to cleaned_labels
	trained_models_dict = train(X_train, y_train)