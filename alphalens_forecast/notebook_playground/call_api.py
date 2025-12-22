#!/usr/bin/env python3                                                                                                                                                                                 
import json                                                                                                                                                                                            
import os                                                                                                                                                                                              
import sys                                                                                                                                                                                             
import urllib.request                                                                                                                                                                                  
import urllib.error                                                                                                                                                                                    
                                                                                                                                                                                                        
                                                                                                                                                                                                        
def post_json(url, payload, timeout=30):                                                                                                                                                               
    data = json.dumps(payload).encode("utf-8")                                                                                                                                                         
    req = urllib.request.Request(                                                                                                                                                                      
        url,                                                                                                                                                                                           
        data=data,                                                                                                                                                                                     
        headers={"Content-Type": "application/json"},                                                                                                                                                  
        method="POST",                                                                                                                                                                                 
    )                                                                                                                                                                                                  
    try:                                                                                                                                                                                               
        with urllib.request.urlopen(req, timeout=timeout) as resp:                                                                                                                                     
            body = resp.read().decode("utf-8")                                                                                                                                                         
            return json.loads(body)                                                                                                                                                                    
    except urllib.error.HTTPError as exc:                                                                                                                                                              
        body = exc.read().decode("utf-8")                                                                                                                                                              
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc                                                                                                                                        
                                                                                                                                                                                                        
                                                                                                                                                                                                        
def main():                                                                                                                                                                                            
    api_url = os.getenv("ALPHALENS_API_URL", "http://3.17.224.165:8000/forecast")                                                                                                                      
                                                                                                                                                                                                        
    payload = {                                                                                                                                                                                        
        "symbol": "USD/JPY",                                                                                                                                                                           
        "timeframe": "15min",                                                                                                                                                                          
        "horizons": [1, 3, 6],                                                                                                                                                                         
        "use_montecarlo": True,                                                                                                                                                                        
        "paths": 3000,                                                                                                                                                                                 
        "include_predictions": True,                                                                                                                                                                   
        "include_metadata": True,                                                                                                                                                                      
        "include_model_info": True,                                                                                                                                                                    
    }                                                                                                                                                                                                  
                                                                                                                                                                                                        
    resp = post_json(api_url, payload)                                                                                                                                                                 
    if not resp.get("ok", False):                                                                                                                                                                      
        print(json.dumps(resp, indent=2))                                                                                                                                                              
        sys.exit(1)                                                                                                                                                                                    
                                                                                                                                                                                                        
    data = resp.get("data", {})                                                                                                                                                                        
    print("ok:", resp.get("ok"))                                                                                                                                                                       
    print("\npayload (resume horizons):")                                                                                                                                                              
    print(json.dumps(data.get("payload", {}), indent=2))                                                                                                                                               
                                                                                                                                                                                                        
    print("\npredictions (series):")                                                                                                                                                                   
    print(json.dumps(data.get("predictions", {}), indent=2))                                                                                                                                           
                                                                                                                                                                                                        
                                                                                                                                                                                                        
if __name__ == "__main__":                                                                                                                                                                             
    main()                      




#  #!/usr/bin/env python3
#   import argparse
#   import json
#   import os
#   import sys
#   import urllib.error
#   import urllib.request


#   #!/usr/bin/env python3
#   import argparse
#   import json
#   import os
#   import sys
#   import urllib.error
#   import urllib.request

#   #!/usr/bin/env python3
#   import argparse
#   import json
#   import os
#   import sys
#   import urllib.error
#   import urllib.request


#   def post_json(url, payload, timeout=30):
#       data = json.dumps(payload).encode("utf-8")
#       req = urllib.request.Request(
#           url,
#           data=data,
#   import sys
#   import urllib.error
#   import urllib.request


#   import urllib.request


#       req = urllib.request.Request(
#           url,
#           data=data,
#           headers={"Content-Type": "application/json"},
#           method="POST",
#       )
#       try:
#           with urllib.request.urlopen(req, timeout=timeout) as resp:
#               body = resp.read().decode("utf-8")
#               return json.loads(body)
#       except urllib.error.HTTPError as exc:
#           body = exc.read().decode("utf-8")
#           raise RuntimeError(f"HTTP {exc.code}: {body}") from exc



#   def parse_args():
#       parser = argparse.ArgumentParser(description="Call AlphaLens inference   
#   API")
#       parser.add_argument(
#           "--api-url",
#           default=os.getenv("ALPHALENS_API_URL", "http://3.17.224.165:8000/    
#   forecast"),
#           help="API endpoint URL",
#       )                                                                        
#       parser.add_argument("--symbol", default="XAU/USD")                       
#       parser.add_argument("--timeframe", default="15min")                      
#       parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6])      parser.add_argument("--paths", type=int, default=3000)                   
                                                                               
#       parser.add_argument("--montecarlo", dest="use_montecarlo",               
#   action="store_true")                                                         
#       parser.add_argument("--no-montecarlo", dest="use_montecarlo",            
#   action="store_false")                                                        
#       parser.set_defaults(use_montecarlo=True)                                 
                                                                               
#       parser.add_argument("--include-predictions", action="store_true")        
#       parser.add_argument("--include-metadata", action="store_true")           
#       parser.add_argument("--include-model-info", action="store_true")         
#       parser.add_argument("--raw", action="store_true", help="Print full JSON  
#   response")                                                                   
#       return parser.parse_args()                                               
                                                                               
                                                                               
#   def main():                                                                  
#       args = parse_args()                                                      
#       payload = {                                                              
#           "symbol": args.symbol,                                               
#           "timeframe": args.timeframe,                                         
#           "horizons": args.horizons,                                           
#           "use_montecarlo": args.use_montecarlo,                               
#           "paths": args.paths,                                                 
#           "include_predictions": args.include_predictions,                     
#           "include_metadata": args.include_metadata,                           
#           "include_model_info": args.include_model_info,                       
#       }                                                                        
                                                                               
#       resp = post_json(args.api_url, payload)                                  
#       if not resp.get("ok", False):                                            
#           print(json.dumps(resp, indent=2))                                    
#           sys.exit(1)                                                          
                                                                               
#       if args.raw:                                                             
#           print(json.dumps(resp, indent=2))                                    
#           return                                                               
                                                                               
#       data = resp.get("data", {})                                              
#       print("payload:")                                                        
#       print(json.dumps(data.get("payload", {}), indent=2))                     
                                                                               
#       preds = data.get("predictions")                                          
#       if preds is not None:                                                    
#           print("\npredictions:")                                              
#           print(json.dumps(preds, indent=2))                                   
#       else:                                                                    
#           print("\npredictions: not included (use --include-predictions)")     
                                                                               
#       metadata = data.get("metadata")                                          
#       if metadata is not None:                                                 
#           print("\nmetadata:")                                                 
#           print(json.dumps(metadata, indent=2))                                
                                                                               
                                                                               
#   if __name__ == "__main__":                                                   
#       main()                                                                   
                  