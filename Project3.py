from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize


def load_data(fname):
	document_data = []
	label data = []
	with open(fname, 'r') as file:
		for record in file:
			split_record = record.split(',')
			label_dataa.append(split_record[0])
			document_data.append(split_record[5])

def clean_data(documents, labels):
	"""
	labels: [1 if label else 0 for label in labels]

	documents:
		drop all but last column
		get vocab and word counts
		drop all words with less than 10 occurrances
		drop top 15 words (or nltk stopwords)
		drop all words with @
		drop all words with http
	"""
	cleaned_labels = [1 if label else 0 for label in labels]
	pass

def train_doc2vec(cleaned_documents):
	"""
	tokenize cleaned dataset using nltk
	train doc2vec from gensim
	save model to disk
	return model
	"""
	fname = "doc2vec"
	model = Doc2Vec(cleaned_documents)
	model.save(fname)
	model = Doc2Vec.load(fname)
	pass

def tokenize_data(cleaned_documents, d2v_model):
	"""
	use "infer_vector" from d2v_model

	return vectorized_docs -> list: vectorized set of documents 
	"""
	return d2v_model.infer_vector(cleaned_documents)

def train(X_train, y_train):
	"""
	train multiple models from scikit-learn

	return models: {"model": model_object, etc}
	"""
	pass

def test(trained_models_dict, X_test, y_test):
	pass