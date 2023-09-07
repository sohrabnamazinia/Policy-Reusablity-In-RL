import psycopg2

class amazonDB:

    def __init__(self, database="amazon_reviews", user="postgres", passwd="2534", host="localhost", port="5432"):
        self.connection = psycopg2.connect(database=database, user=user, password=passwd, host=host, port=port)
        self.cursor = self.connection.cursor()
    
    def get_reviews(self):
        
        self.cursor.execute("SELECT reviewtext FROM reviews")
        rows = self.cursor.fetchall()
        reviews = []
        
        for row in rows:
            reviews.append(row)

        self.cursor.close()
        self.connection.close()

        return reviews






    