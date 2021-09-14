from flask import Flask, request, jsonify, json
from DeepKnowledgeTracing.QR import qr

import csv
app = Flask(__name__)

# cors = CORS(app)


@app.route('/qr', methods=['POST'])
def start():

    records = request.form
    records = json.loads(list(records)[0])["records"]
    with open('DeepKnowledgeTracing/data/a.csv', "r") as f:
        rows = []
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    question_list = []
    correct_list = []
    for i in range(len(records)):
        index = rows[0].index(str(records[i][0]))
        question_list.append(rows[1][index])
        correct_list.append(str(records[i][1]))
    input = []
    students = []
    students.append("0")
    students.append(question_list)
    students.append(correct_list)
    input.append(students)
    target, tr = qr(input)
    print(target)
    for i in range(len(rows[1])):
        if rows[1][i] == str(target):
            target = rows[0][i]
            break
    print(target)
    return jsonify({"target": int(target)})


if __name__ == '__main__':
    app.run()
