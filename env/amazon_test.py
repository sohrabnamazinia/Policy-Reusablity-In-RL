from amazonDB import amazonDB

amazon_db = amazonDB()
reviews = amazon_db.get_reviews()
print(reviews)