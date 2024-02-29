class ModelEncoder:
    def __init__(self, model, is_cuda):
        self.model = model
        self.is_cuda = is_cuda

    def encode_param(self, param):
        dumps = param.tobytes()
        return dumps

    def decode_param(self, data):
        dumps = data
        arr = np.frombuffer(dumps, dtype=np.float32)
        return arr

    def encode(self):
        model = self.model.state_dict()
        total_size = 0
        s = time.time()
        serialized = []
        serialized.append(struct.pack('i', len(model.keys())))
        pciet = 0.
        for name, param in model.items():
            tmpt = time.time()
            ny = param.cpu().numpy()
            pciet += time.time() - tmpt
            #ny = param.cpu().numpy().astype(np.float16)
            byteparam = self.encode_param(ny)
            bytename = name.encode('utf-8')
            serialized.append(struct.pack('i', len(bytename)))
            #logger.debug('encode name l: %d', len(name))
            serialized.append(bytename)
            serialized.append(struct.pack('i', len(byteparam)))
            #logger.debug('encode model l: %d', len(byteparam))
            serialized.append(byteparam)
            total_size += ny.size
        serialized = b''.join(serialized)
        logger.debug('model total size: %d, --get model time used: %f, pcie time: %f', total_size * 4, time.time()-s, pciet)
        return serialized 

    def decode(self, encoded_model):
        serialized = encoded_model
        own_state = {}
        offset = 0
        num_item = struct.unpack('i', serialized[offset:offset+4])[0]
        offset += 4
        for i in range(num_item):
            l = struct.unpack('i', serialized[offset:offset+4])[0]
            #logger.debug('decode name l: %d', l)
            offset += 4
            name = serialized[offset:offset+l].decode('utf-8')
            offset += l
            l = struct.unpack('i', serialized[offset:offset+4])[0]
            #logger.debug('decode model l: %d', l)
            offset += 4
            param = serialized[offset:offset+l]
            offset += l
            own_state[name] = param
        return own_state

    def param_average(self, a, b, ratio, is_asked):
        if self.is_cuda:
            a_tensor = a.cuda()
            b_tensor = torch.from_numpy(b).cuda()
        else:
            a_tensor = a.cpu() 
            b_tensor = torch.from_numpy(b)
        new_param = (a_tensor+b_tensor.view(a_tensor.size()))/2.0
        return new_param

    def average_model(self, model, recved_loss, is_asked):
        own_state = self.model.state_dict()
        s = time.time()
        loss = 1.0#self.get_loss()
        average_ratio = 1.0
        if recved_loss > 0:
            r = (recved_loss - loss) / recved_loss
            if r > 0.2:
                average_ratio = 1.001#1+self.lr 
            elif r < -0.2:
                average_ratio = 0.999#1-self.lr 
            else:
                average_ratio = 1
        recv_state = self.decode(model)
        for name, param in own_state.items():
            remote_param = self.decode_param(recv_state[name])  
            new_param = self.param_average(param, remote_param, average_ratio, is_asked) 
            own_state[name] = new_param
        self.model.load_state_dict(own_state)
        if self.is_cuda:
            self.model.cuda()
        logger.debug('====model average time: %f', time.time()-s)

    def get_model(self):
        return self.encode()

