from flask import Flask, request, make_response,jsonify,Response


app = Flask(__name__)

def createJsonResponse(code,result):
     data = {}
    #  resp.status_code = code
     data["result"] = result
     if code == 200:
        data["success"] = True
     else:
        data["success"] = False
     response = make_response(data, code)
     response.headers["Content-Type"] = 'application/json'
     return response

@app.route("/")
def hello():
    return "pet face flask python tuzi"



   


if __name__ == "__main__":
   

    #model = torch.hub.load('ikouhaha/pet_face_yolov5','custom', path='best.pt', force_reload=True)  # force_reload to recache
    app.run(host='0.0.0.0',port=5000)  # debug=True causes Restarting with stat
