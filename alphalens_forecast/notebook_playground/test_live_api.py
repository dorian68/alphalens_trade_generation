#%%
import requests                                                                     
                                                                                    
url = "http://3.142.243.85:8000/forecast"                                        
payload = {                                                                         
    "symbol": "BTC/USD",                                                            
    "timeframe": "15min",                                                           
    "horizons": [1, 3, 6],                                                          
    "use_montecarlo": True,                                                         
    "paths": 3000,                                                                  
}                                                                                   
                                                                                    
resp = requests.post(url, json=payload, timeout=60)                                 
print(resp.status_code)                                                             
print(resp.json())                                                                  
                                                                                      
#   Navigateur                                                                          
                                                                                      
#   - Pour vérifier que le serveur répond :                                             
#     http://<EC2_PUBLIC_IP>:8000/health                                                
#   - Pour lancer un forecast (POST), ouvre la console (F12) et exécute :
                                                                                      
#   fetch("http://<EC2_PUBLIC_IP>:8000/forecast", {                                     
#     method: "POST",                                                                   
#     headers: { "Content-Type": "application/json" },                                  
#     body: JSON.stringify({                                                            
#       symbol: "BTC/USD",                                                              
#       timeframe: "15min",                                                             
#       horizons: [1, 3, 6],                                                            
#       use_montecarlo: true,                                                           
#       paths: 3000                                                                     
#     })                                                                                
#   })                                                                                  
#   .then(r => r.json())                                                                
#   .then(console.log)                                                                  
#   .catch(console.error);                                                              
                                                                                      
#   Note : si tu lances ça depuis une page d’un autre domaine, le navigateur peut       
#   bloquer la requête (CORS). Dans ce cas, utilise curl/Postman, ou teste depuis un    
#   onglet ouvert directement sur l’IP de ton EC2.  