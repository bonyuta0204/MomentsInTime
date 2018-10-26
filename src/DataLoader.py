import MITDataLoader
import MITData
import net from Network

result_index = []
# load data and give them to CNN
for index, data in MITDataLoader:
    # image batch is tensor with shape (N, 3, x, y)
    # predict has shape (N, n_classes)
    predict = net(data.image_batch)
    predict_ave = predict.mean(axis=1)
    label = torch.argmax(predict_ave)
    result_index.append({"index":index,
        "directory": data.directory,
        "train":data.train,
        "category":data.category
        "object-category":label})

