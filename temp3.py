import requests

ids = [
        "42d704df71994c52e1f8ed58b0b79793",
        "f12269362d7568900328620a8a5c2daa",
        "75231084abb6f351d5df16fe0333f287",
        "ee21a00d5fc1c7c51cb8c0a582b360b0",
        "65181588ca03ba9bf3f36be114071794",
        "c713c2828dc2dd727b6455e34bce29d9",
        "c4002590b2d4df9f571fba3cfca64b49",
        "5f2b96b73a9a6f229dfe112498d76d21",
        "b35387bc6d511372d2521f2a1d0d85aa",
        "f38f87723dea73d09be6c420f12903f8",
        "2o24O773b75a406abe3a19e9391a3b23",
        "2o24O7785335438499cce93fb73091c1",
        "2o24O75a71ea49bf870f43208a6ac993",
        "2o24O70205f4425c84017e6685cde21a",
        "2o24O71174264b85810f512220522f42",
        "2o24O7d0200a4b3691b58eedaa394c2d",
        "2o24O74275564474bc0274acde7253ea",
        "2o24O7466253431b85b9bba86e2a443f",
        "2o24O711992243f4a02a2894e0b0a7b2",
        "2o24O7e3ac0f46539617903a1ff72e67",
        "2o24O7fe6887455f990fde4dec7bd92a",
        "2o24O7eb221f456a8de0e5913431e5b0",
        "2o24O77274ec4a678707c0809538db19",
        "2o24O728f6794d5c936cf773fdc403e2",
        "2o24O7b9fa7544b6aebac841583f3ab4",
        "2o24O747e80f4a3eb2cc61e4ac446356",
        "2o24O762f8354a49bb9bc22a153d0eba",
        "2o24O7b583984ee0b3e6c2d5f1c6dad1",
        "2o24O78d9e8f44d8af908e3d4c783b10",
        "2o24O7b4ac9e4305a15380cf61a8f3f9",
        "c2fcab02ab871a8eb9d1a4b02e6b4e0c",
        "14fd55f6ac77d8591b0c6ab2e18bdc22",
        "181dbf6f0d51094d74e4282e01849aa5",
        "f1a06718eb48db83943953ea86c754a6",
        "6079aaae268483b1bf61354a8f5bb238",
        "74c4c4a5a5d339ca38e1ab1a1110e531",
        "de9c83a7085eef11b2a464ec3402425b",
        "db9dc9f93796e953a42bca076cee7707",
        "2o24O70bea57484c84b0ca006e749ce0",
        "e2119fb0af81aa92cb2d5ac9419fda72",
        "c72e4c3079435243b73f9fafb9c77eb4",
        "58350b4fd3afa57d6660056bca7aa3c9",
        "51f1d4975ac30ae1ff567dfe4f3f6b4c",
        "8aa028e1fab8eed41463025684d898dc",
        "02fd7910de3dc12cc7e298c4dd6b6904",
        "cd18dd9c1f3de3999c4988e5e9c84861",
        "19570f7b3667b8b1c4e197157309ded9",
        "717e1a29ac4b603c6ee2d4933edf07e2",
        "20c5f0ceb420c8a30ca9fa8ea2fbd845",
        "39738aa18e253a46f9654595086b8363"
    ]
header = {
    "token": "363b2fab127062be52b391dcfad7bb38"
}


url = "http://regi.zju.edu.cn/grs-pro/config/questions/getExamsByIds"

res = requests.post(url, json=ids, headers=header).json()
print(res["exams"])

i = 1
for exam in res["exams"]:
    answerIds = exam["answer"].split(",")
    answerList = exam["answers"]
    foundAnswers = []
    foundAnswersNum = []
    for answer in answerList:
        if answer["id"] in answerIds:
            foundAnswers.append(answer["content"])
            foundAnswersNum.append(answer["orderNum"])
        if answer["isAnswer"] == "1":
            foundAnswers.append(answer["content"])
            foundAnswersNum.append(answer["orderNum"])

    if foundAnswers:
        print("序号：%d, 答案：%s  %s" % (i, "，".join(foundAnswersNum), "，".join(foundAnswers)))
        i += 1