import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import nmslib
from flask import jsonify,render_template, send_from_directory
from flask import request
from flask import Flask
import cleantxt
import logging
import math
import json
 
print('loading model')
model =  gensim.models.Word2Vec.load('full_grams_cbow_300.mdl')
model.clear_sims()
print('model loaded')




print('loading sugesstion data')
#data = pd.DataFrame(columns={'SUG_ID','SUG_TITLE','SUG_BODY1'})
data =  pd.read_csv('sug.csv',  error_bad_lines=False,encoding='utf-8', lineterminator='\n',warn_bad_lines=False ,sep ='|',
                   header=0 )
data.replace(to_replace='\r',value='', regex=True)
data['title_vec'] = data['TITLE'].apply(lambda x:  cleantxt.calc_vec(pos_tokens=gensim.utils.simple_preprocess(cleantxt.clean_str(x)),
    neg_tokens=[''], n_model=model, dim=model.vector_size)  )
data_val = data["title_vec"]
data_id = data["S_ID"]
data_val = data_val.values
data_id = data_id.values
data_val_n = np.stack( data_val, axis=0 )

print('sugesstion data loaded')

print('loading index') 

# initialize a new index, using a HNSW index on Cosine Similarity
index = nmslib.init(method='hnsw', space='cosinesimil', dtype = nmslib.DistType.DOUBLE )
index.addDataPointBatch(data_val_n,data_id)
index.loadIndex('sug_index')
efS = 100
query_time_params = {'efSearch': efS}
index.setQueryTimeParams(query_time_params)
print('index loaded')

app = Flask(__name__)

@app.route('/search', methods=['GET'])
def search(): 
    pos_srch = request.args.get('pos_srch')        
    #neg_srch = request.args.get('neg_srch')        
    
    pos_srch = cleantxt.clean_str(pos_srch)
    pos_srch = gensim.utils.simple_preprocess(pos_srch)
    
    #neg_srch = cleantxt.clean_str(neg_srch)
    #neg_srch = gensim.utils.simple_preprocess(neg_srch)
    neg_srch=['']
    srch =  cleantxt.calc_vec(pos_tokens=pos_srch, neg_tokens=neg_srch, n_model=model, dim=model.vector_size)
    result = []    
    # query for the nearest neighbours of the first datapoint
    ids, distances = index.knnQuery(srch, k=5)
    for id,dis in zip(ids,distances):
      result.append([ str( int(100*(1-dis) ))+'%'  , int(data.loc[data['S_ID'] ==  id]['S_ID'].values[0] ),
                     data.loc[data['S_ID'] ==  id]['TITLE'].values[0] ])  
    print(result)  
    return  json.dumps(result, ensure_ascii=False)

@app.route('/add', methods=['GET'])
def add(): 
    sug_id = int (request.args.get('sug_id') )
    sig_title=  request.args.get('sig_title') 
    #neg_srch = request.args.get('neg_srch')        
    
    sig_title_vec = cleantxt.clean_str(sig_title)
    sig_title_vec = gensim.utils.simple_preprocess(sig_title)
    sig_title_vec =  cleantxt.calc_train_vec(pos_tokens=sig_title_vec, n_model=model, dim=model.vector_size)
    model.clear_sims()
    index.addDataPoint(sug_id , sig_title_vec)
    M = 70
    efC = 300
    num_threads = 4
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 2}
    index.createIndex(index_time_params, print_progress=True)
    return  json.dumps("done", ensure_ascii=False)



if __name__ == "__main__":
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print ("Starting server on http://localhost:5000")
    print ("Serving ...",  app.run(host='0.0.0.0'))
    print ("Finished !")
    print ("Done !")