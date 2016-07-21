import artm
import uuid
import copy
import numpy as np


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
        self._create_parent_phi_batch()
        self._create_parent_phi_batch_vectorizer()
        
        
    def _create_parent_phi_batch(self):
        """
        :description: creates new batch with parent level topics as documents   
        """
        batch_dict = {}
        NNZ = 0
        batch = artm.messages.Batch()
        batch.id = str(uuid.uuid4())
        batch_id = batch.id
        batch.description = "__parent_phi_matrix_batch__"
        for topic_name_idx in range(len(self.parent_model.topic_names)):
            topic_name = self.parent_model.topic_names[topic_name_idx]
            phi = self.parent_model.get_phi(topic_names={topic_name})
            if not topic_name_idx:
                # batch.token is the same for all topics, create it once
                topics_and_tokens_info = \
                      self.parent_model.master.get_phi_info(self.model_pwt)
                for token, class_id in \
                      zip(topics_and_tokens_info.token, topics_and_tokens_info.class_id):
                    if token not in batch_dict:
                        batch.token.append(token)
                        batch.class_id.append(class_id)
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
            
            
    def _create_parent_phi_batch_vectorizer(self):
        self.phi_batch_vectorizer = artm.BatchVectorizer(batches=[''])
        self.phi_batch_vectorizer._batches_list[0] = artm.batches_utils.Batch(
                                                      self.phi_batch_path)
        self.phi_batch_vectorizer._weights = [self.phi_batch_weight]


    def fit_offline(self, batch_vectorizer, num_collection_passes=1, *args, **kwargs):
        modified_batch_vectorizer = copy.copy(batch_vectorizer)
        modified_batch_vectorizer._batches_list.append(
                artm.batches_utils.Batch(self.phi_batch_path))
        modified_batch_vectorizer._weights.append(self.phi_batch_weight)
        #import_batches_args = artm.wrapper.messages_pb2.ImportBatchesArgs(
        #                           batch=[self.parent_batch])
        #self._lib.ArtmImportBatches(self.master.master_id, import_batches_args)
        hierarchy_sparsing_regularizer_names = set()
        for reg_name in self.regularizers.data:
            if isinstance(self.regularizers[reg_name], artm.regularizers.HierarchySparsingThetaRegularizer):
                hierarchy_sparsing_regularizer_names.add(reg_name)
                if not len(self.regularizers[reg_name].config.topic_proportion):
                    for _ in xrange(self.num_topics):
                        self.regularizers[reg_name].config.topic_proportion.append(1.0)
                if not len(self.regularizers[reg_name].config.parent_topic_proportion):
                    for _ in xrange(self.parent_model.num_topics):
                        self.regularizers[reg_name].config.parent_topic_proportion.append(1.0)
                if len(self.regularizers[reg_name].config.topic_proportion) != self.num_topics:
                    raise ValueError("len(regularizers["+reg_name+"].config.topic_proportion) != num_topics")
                if len(self.regularizers[reg_name].config.parent_topic_proportion) != self.parent_model.num_topics:
                    raise ValueError("len(regularizers["+reg_name+"].config.parent_topic_proportion) != parent_model.num_topics")
        if not len(hierarchy_sparsing_regularizer_names):
            super(ARTM_Level, self).fit_offline(modified_batch_vectorizer, *args, **kwargs)
        else:
            for _ in xrange(num_collection_passes):
                psi = self.get_psi(ignore_cache_theta=True).values 
                for reg_name in hierarchy_sparsing_regularizer_names:
                    parent_topic_proportion = np.array(self.regularizers[reg_name].config.parent_topic_proportion)
                    topic_proportion = (psi * parent_topic_proportion[np.newaxis, :]).sum(axis=1)
                    for topic_idx in xrange(self.num_topics):
                        self.regularizers[reg_name].config.topic_proportion[topic_idx] = float(topic_proportion[topic_idx])
                    self._master.reconfigure_regularizer(reg_name, self.regularizers[reg_name].config)
                super(ARTM_Level, self).fit_offline(modified_batch_vectorizer, num_collection_passes=1)
                
                
    def fit_online(self, *args, **kwargs):
        raise NotImplementedError("Fit_online method is not implemented in hARTM. Use fit_offline method.")


    def get_psi(self, ignore_cache_theta=False):
        """
        :returns: p(subtopic|topic) matrix
        
        :param ignore_cache_theta: if True, get_theta() method will not be used
        """
        if self.cache_theta and not ignore_cache_theta:
            current_columns_naming = self.theta_columns_naming
            self.theta_columns_naming = "title"
            psi = self.get_theta()[self.parent_model.topic_names]
            self.theta_columns_naming = current_columns_naming
        else:
            current_columns_naming = self.theta_columns_naming
            self.theta_columns_naming = "title"
            psi = self.transform(self.phi_batch_vectorizer)
            self.theta_columns_naming = current_columns_naming
        return psi