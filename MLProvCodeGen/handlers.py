import json

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado
from . import app

class RouteHandler(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def get(self):
        self.finish(json.dumps({
            "data": "/MLProvCodeGen/get_example endpoint!"
        }))

    @tornado.web.authenticated
    def post(self):
        # input_data is a dictionary with a key "name"
        input_data = self.get_json_body()
        data = {"data": "/{}/post_example endpoint!".format(input_data["name"])}
        self.finish(json.dumps(data))
   
class RouteHandler2(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        # input_data is a dictionary with a key "name"
        code = self.get_json_body()
        code2 = app.app(code)
        data = {"greetings": "LISTEN HERE, LISTEN WELL"}
        self.finish(json.dumps(code))

class RouteHandler3(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        #input_data is an object called objBody
        input_data = self.get_json_body()
        reply = app.IC_pytorch(input_data)
        self.finish(json.dumps(reply))

class RouteHandler4(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        #input_data is an object called objBody
        input_data = self.get_json_body()
        reply = app.IC_scikit(input_data)
        self.finish(json.dumps(reply))

class RouteHandler5(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        #input_data is an object called objBody
        input_data = self.get_json_body()
        reply = app.Clustering_scikit(input_data)
        self.finish(json.dumps(reply))

class RouteHandler6(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        #input_data is an object called objBody
        input_data = self.get_json_body()
        reply = app.MS_scikit(input_data)
        self.finish(json.dumps(reply))        


class RouteHandler7(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        #input_data is an object called objBody
        input_data = self.get_json_body()
        reply = app.MulticlassClassification(input_data)
        self.finish(json.dumps(reply))  

def setup_handlers(web_app):
    host_pattern = ".*$"

    base_url = web_app.settings["base_url"]
    route_pattern = url_path_join(base_url, "MLProvCodeGen", "get_example")
    route_pattern2 = url_path_join(base_url, "MLProvCodeGen", "post_example")
    route_pattern3 = url_path_join(base_url, "MLProvCodeGen", "ImageClassification_pytorch")
    route_pattern4 = url_path_join(base_url, "MLProvCodeGen", "ImageClassification_scikit")
    route_pattern5 = url_path_join(base_url, "MLProvCodeGen", "Clustering_scikit")
    route_pattern6 = url_path_join(base_url, "MLProvCodeGen", "ModelSelection_scikit")
    route_pattern7 = url_path_join(base_url, "MLProvCodeGen", "MulticlassClassification")
    handlers = [(route_pattern, RouteHandler),
                (route_pattern2, RouteHandler2),
                (route_pattern3, RouteHandler3),
                (route_pattern4, RouteHandler4),
                (route_pattern5, RouteHandler5),
                (route_pattern6, RouteHandler6),
                (route_pattern7, RouteHandler7)]
   
    web_app.add_handlers(host_pattern, handlers)
