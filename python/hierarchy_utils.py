import artm
import uuid
import copy


class ARTM_Level(artm.ARTM):
    def __init__(self, parent_model, phi_batch_weight=1, phi_batch_path="", 
                 *args, **kwargs):
        """
        :description: builds one hierarchy level that is usual topic model
        
        :param parent_model: ARTM or ARTM_Level instance, previous level model,
         already built
        :param phi_batch_weight: float, weight of parent phi batch
        :param phi_batch_path: string, path where to save phi parent batch,
         default '', temporary solution
        :other params as in ARTM class
        
        :Notes:
             * Parent phi batch consists of documents that are topics (phi columns) 
               of parent_model. Corresponding to this batch Theta part is psi matrix
               with p(subtopic|topic) elements.
             * To get psi matrix use get_psi() method.
               Other methods are as in ARTM class.
        """
        self.parent_model = parent_model
        self.phi_batch_weight = phi_batch_weight
        self.phi_batch_path = phi_batch_path + "phi.batch"
        super(ARTM_Level, self).__init__(*args, **kwargs)
        self.create_parent_phi_batch()
        
        
    def create_parent_phi_batch(self):
        """
        :description: creates new batch with parent level topics as documents   
        """
        batch_dict = {}
        NNZ = 0
        batch = artm.messages.Batch()
        batch.id = str(uuid.uuid4())
        batch_id = batch.id
        for topic_name_idx in range(len(self.parent_model.topic_names)):
            topic_name = self.parent_model.topic_names[topic_name_idx]
            phi = self.parent_model.get_phi(topic_names={topic_name})
            if not topic_name_idx:
                #batch.token is the same for all topics, create it once
                for token in phi.index:
                    if token not in batch_dict:
                        batch.token.append(token)
                        batch_dict[token] = len(batch.token) - 1
            
            # add item (topic) to batch
            item = batch.item.add()
            item.title = topic_name
            field = item.field.add()
            indices = phi[topic_name] > 0
            for token, weight in  \
                     zip(phi.index[indices], phi[topic_name][indices]):
                    field.token_id.append(batch_dict[token])
                    field.token_weight.append(float(weight))
                    NNZ += weight
        #self.parent_batch = batch
        with open(self.phi_batch_path, 'wb') as fout:
            fout.write(batch.SerializeToString())
        
    
    def fit_offline(self, batch_vectorizer, *args, **kwargs):
        modified_batch_vectorizer = copy.deepcopy(batch_vectorizer)
        modified_batch_vectorizer._batches_list.append(
                artm.batches_utils.Batch(self.phi_batch_path))
        modified_batch_vectorizer._weights.append(self.phi_batch_weight)
        #import_batches_args = artm.wrapper.messages_pb2.ImportBatchesArgs(
        #                           batch=[self.parent_batch])
        #self._lib.ArtmImportBatches(self.master.master_id, import_batches_args)
        super(ARTM_Level, self).fit_offline(modified_batch_vectorizer, *args, **kwargs)
        
        
    def get_psi(self):
        if self.cache_theta:
            current_columns_naming = self.theta_columns_naming
            self.theta_columns_naming = "title"
            psi = self.get_theta()[self.parent_model.topic_names]
            self.theta_columns_naming = current_columns_naming
        else:
            phi_batch_vectorizer = artm.BatchVectorizer(batches=[''])
            phi_batch_vectorizer._batches_list[0] = artm.batches_utils.Batch(
                                                          self.phi_batch_path)
            phi_batch_vectorizer._weights = [self.phi_batch_weight]
            current_columns_naming = self.theta_columns_naming
            self.theta_columns_naming = "title"
            psi = self.transform(phi_batch_vectorizer)
            self.theta_columns_naming = current_columns_naming
        return psi    