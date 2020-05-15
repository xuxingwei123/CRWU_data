
import logging
import os
import numpy as np
import tensorflow as tf
import config
import lstm_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


FLAGS = config.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



def report_accuracy(logist, label):
    logist = np.reshape(logist,[-1])
    label = np.reshape(label,[-1])
    average = np.sum(label)/len(label)
    ssr = 0
    sst = 0
    for i in range(len(logist)):
        ssr = ssr+pow((label[i]-logist[i]),2)
        sst = sst+pow((label[i]-average),2)
    print("模型泛用性为：",1-(ssr/sst))
    return 1-(ssr/sst)


def plot_confusion(label, confusion, real_label):
    classes = list(set(label))
    print(classes)

    classes.sort()
    fig,ax=plt.subplots()
    #ax.set(ylabel= 'Predicted values',xlabel='Actual values')
    font_format ={'family':'Times New Roman','size':14}
    plt.ylabel('Predicted values',font_format)
    plt.xlabel('Actual values',font_format)
    plt.imshow(confusion, cmap=plt.cm.rainbow)
    plt.rc('font', family='Times New Roman',size =14)

    ax.set_xticks(np.arange(confusion.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(confusion.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="b", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    indices = range(len(confusion))

    plt.xticks(indices, classes,fontproperties ='Time New Roman',size = 14)
    plt.yticks(indices, classes,fontproperties ='Time New Roman',size = 14)
    plt.colorbar()
   # ax.set(xticks)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index,
                     str(confusion[first_index][second_index]) + '\n' + str(
                         round(confusion[first_index][second_index] / np.sum(real_label == first_index),4)*100)+'%',
                     verticalalignment='center',
                     horizontalalignment='center')
    ax =plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()


    # print(X.shape)




def train(train_data_dir=None, val_data_dir=None,pre_data_dir=None):

    model = lstm_model.LSTMOCR()

    model.build_graph()

    print('loading train data')
    train_feeder = config.DataIterator(data_dir=train_data_dir)

    print('loading validation data')
    val_feeder = config.DataIterator(data_dir=val_data_dir)

    pre_feeder = config.DataIterator(data_dir=pre_data_dir)



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                print("1")
                saver.restore(sess, ckpt)

        if(FLAGS.mode == 'train'):
            print('=============================begin training=============================')
            batch_loss = []
            batch_acc =[]
            for cur_epoch in range(100):
                #print('----------> 第 ',cur_epoch,' 轮 <----------')
                train_y, train_x1, train_x2 = train_feeder.read_data()

                for batch in range(int(len(train_y)/FLAGS.batch_size)):
                    print(int(len(train_y)/FLAGS.batch_size))
                    print('trainy是多少',len(train_y))
                    feed = {model.input_1: train_x1[batch*FLAGS.batch_size:(batch+1)*FLAGS.batch_size],
                            model.input_2: train_x2[batch*FLAGS.batch_size:(batch+1)*FLAGS.batch_size],
                            model.labels:train_y[batch*FLAGS.batch_size:(batch+1)*FLAGS.batch_size]}

                    summary_str, batch_cost, step,learn_rate, confusion,acc, _ = sess.run([model.merged_summay, model.loss, model.global_step ,model.lrn_rate,
                                                                                    model.confusion_matrix,model.acc,model.optimizer], feed)




                    print("第",step,"步：  第",cur_epoch,"轮第",batch,"批数据的损失为：" ,batch_cost, "训练准确率为：",acc)
                    batch_loss.append(batch_cost)
                    batch_acc.append(acc)
                    var = pd.DataFrame(batch_loss)
                    var1 =pd.DataFrame(batch_acc)
                    path_data = './data/'
                    var.to_csv(path_data + '/batch_loss.csv', index=False, header=False)
                    var1.to_csv(path_data + '/acc.csv', index=False, header=False)
                    #plot_result(test_result, test_label1, path)
                    #plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)


                if(cur_epoch%10 == 0):
                    saver.save(sess, "./checkpoint/" + 'model.ckpt', global_step=step)


                if(cur_epoch%2 == 0):
                    test_y, test_x1, test_x2 = val_feeder.read_data()
                    final_acc = 0
                    final_loss = 0
                    for batch in range(int(len(test_y) / FLAGS.batch_size)):
                        feed = {model.input_1: test_x1[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                                model.input_2: test_x2[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                                model.labels: test_y[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]}
                        # 运行图
                        summary_str, batch_cost,logits,acc= sess.run([model.merged_summay, model.loss,model.logits,model.acc], feed)

                        final_acc+=acc
                        final_loss+=batch_cost

                    print("测试的平均损失为：",final_loss/int(len(test_y) / FLAGS.batch_size))
                    print("测试的平均准确率为：", final_acc / int(len(test_y) / FLAGS.batch_size))


        if (FLAGS.mode == 'val'):
            print('=============================begin val=============================')

            test_y, test_x1, test_x2 = val_feeder.read_data()
            final_acc = 0
            final_loss = 0
            for batch in range(int(len(test_y) / FLAGS.batch_size)):
                feed = {model.input_1: test_x1[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                        model.input_2: test_x2[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                        model.labels: test_y[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]}


                summary_str, batch_cost, step, learn_rate, confusion, acc,pre, dense, label,logits1 = sess.run([model.merged_summay,
                                                                                         model.loss,
                                                                                         model.global_step,
                                                                                         model.lrn_rate,
                                                                                         model.confusion_matrix,
                                                                                         model.acc,
                                                                                         model.pre,
                                                                                         model.dense_input,
                                                                                         model.labels,model.logits], feed)
                final_acc += acc
                final_loss += batch_cost

                print("测试的损失为：", batch_cost)
                print(np.sum(test_y[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size] == 0))
                print(np.sum(pre == 0))
                print("测试的准确率为：", acc)


                plot_confusion(pre, confusion,label)

                x = np.concatenate((test_x1[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                                  test_x2[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]),
                                 axis=2)
                x = np.squeeze(x)
                print(x.shape)

                shape = np.shape(x)
                x = x.reshape((shape[0], -1))



                plt.savefig('digits_tsne-generated.png', dpi=120)
                plt.show()


            print("测试的平均准确率为：", final_acc / int(len(test_y) / FLAGS.batch_size))

        if (FLAGS.mode == 'pre'):
            print('=============================begin pre=============================')
            pre_y, pre_x1, pre_x2 = pre_feeder.read_data()
            for batch in range(int(len(train_y) / FLAGS.batch_size)):
                feed = {model.input_1: pre_x1[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                        model.input_2: pre_x2[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size],
                        model.labels: pre_y[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]}
                # 运行图
                summary_str, pre= sess.run([model.merged_summay,model.correct_prediction], feed)

                print("预测结果为：",pre)






if __name__ == '__main__':
    train(FLAGS.train_data_dir, FLAGS.val_data_dir, FLAGS.pre_data_dir)

