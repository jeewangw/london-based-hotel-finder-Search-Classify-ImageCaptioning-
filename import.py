import os
import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

engine = create_engine('mysql+pymysql://root@localhost/usvideos')
db = scoped_session(sessionmaker(bind=engine))

with open("./static/USvideos.csv") as f:
    reader = csv.reader(f, delimiter=',')
    lc = 0
    for row in reader:
        if lc != 0:
            # values, so insert into table
            db.execute('INSERT INTO usvideos ("video_id", "trending_date", "title", "channel_title","category_id", "publish_time","tags","views","likes","dislikes","comment_count","thumbnail_link","comments_disabled","ratings_disabled","video_error_or_removed","description") VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16)',  {
                "1": row[0],
                "2": row[1],
                "3": row[2],
                "4": row[3],
                "5": row[4],
                "6": row[5],
                "7": row[6],
                "8": int(row[7]),
                "9": int(row[8]),
                "10": int(row[9]),
                "11":int(row[10]),
                "12":row[11],
                "13":row[12],
                "14":row[13],
                "15":row[14],
                "16":row[15]
            })

        lc += 1

    db.commit()