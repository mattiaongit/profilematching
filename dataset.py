import pymongo

connection = pymongo.Connection()
db = connection['alternion']

def raw_data():
	return db.profiles.find({},{'_id':0})
