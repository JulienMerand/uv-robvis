import tensorflow as tf 
tf.compat.v1.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plot
import cv2



mnist_train_images=np.fromfile(r"Vision\TP Reseau Neurone\dataset\mnist\train-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 784)/255
mnist_train_labels=np.eye(10)[np.fromfile(r"Vision\TP Reseau Neurone\dataset\mnist\train-labels.idx1-ubyte", dtype=np.uint8)[8:]]
mnist_test_images=np.fromfile(r"Vision\TP Reseau Neurone\dataset\mnist\t10k-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 784)/255
mnist_test_labels=np.eye(10)[np.fromfile(r"Vision\TP Reseau Neurone\dataset\mnist\t10k-labels.idx1-ubyte", dtype=np.uint8)[8:]]

# print("Train dataset length : ", len(mnist_train_images))
# print("Test dataset length : ", len(mnist_test_images))


images = mnist_train_images[:10]
labels = mnist_train_labels[:10]

# import matplotlib for visualization
import numpy as np
import matplotlib.pyplot as plt

# for index, image in enumerate(images):
#     print("Label:",labels[index])
#     print("Digit in the image", np.argmax(labels[index]))
#     plt.imshow(image.reshape(28,28),cmap='gray')
#     plt.show()

ph_images=tf.compat.v1.placeholder(shape=(None, 784), dtype=tf.float32)
ph_labels=tf.compat.v1.placeholder(shape=(None, 10), dtype=tf.float32)

nbr_n1=128
nbr_n2=64
learning_rate=0.0001
taille_batch=100
nbr_entrainement=200

wci=tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape=(784, nbr_n1)), dtype=tf.compat.v1.float32)
bci=tf.compat.v1.Variable(np.zeros(shape=(nbr_n1)), dtype=tf.compat.v1.float32)
sci=tf.compat.v1.matmul(ph_images, wci)+bci
sci=tf.compat.v1.nn.sigmoid(sci)

wci=tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape=(nbr_n1, nbr_n2)), dtype=tf.compat.v1.float32)
bci=tf.compat.v1.Variable(np.zeros(shape=(nbr_n2)), dtype=tf.compat.v1.float32)
sci=tf.compat.v1.matmul(sci, wci)+bci
sci=tf.compat.v1.nn.sigmoid(sci)

wco=tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape=(nbr_n2, 10)), dtype=tf.compat.v1.float32)
bco=tf.compat.v1.Variable(np.zeros(shape=(10)), dtype=tf.compat.v1.float32)
sco=tf.compat.v1.matmul(sci, wco)+bco
scoo=tf.compat.v1.nn.softmax(sco)

loss=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=sco)
train=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy=tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(tf.compat.v1.argmax(scoo, 1), tf.compat.v1.argmax(ph_labels, 1)), dtype=tf.compat.v1.float32))

with tf.compat.v1.Session() as s:
    
    # Initialisation des variables
    s.run(tf.compat.v1.global_variables_initializer())

    tab_acc_train=[]
    tab_acc_test=[]
    
    for id_entrainement in range(nbr_entrainement):
        print("ID entrainement", id_entrainement)
        for batch in range(0, len(mnist_train_images), taille_batch):
            # lancement de l'apprentissage en passant la commande "train". feed_dict est l'option désignant ce qui est
            # placé dans les placeholders
            s.run(train, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })

        # Prédiction du modèle sur les batchs du dataset de training
        tab_acc=[]
        for batch in range(0, len(mnist_train_images), taille_batch):
            # lancement de la prédiction en passant la commande "accuracy". feed_dict est l'option désignant ce qui est
            # placé dans les placeholders
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_train_images[batch:batch+taille_batch],
                ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })
            # création le tableau des accuracies
            tab_acc.append(acc)
        
        # calcul de la moyenne des accuracies 
        print("accuracy train:", np.mean(tab_acc))
        tab_acc_train.append(1-np.mean(tab_acc))
        
        # Prédiction du modèle sur les batchs du dataset de test
        tab_acc=[]
        for batch in range(0, len(mnist_test_images), taille_batch):
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_test_images[batch:batch+taille_batch],
                ph_labels: mnist_test_labels[batch:batch+taille_batch]
            })
            tab_acc.append(acc)
        print("accuracy test :", np.mean(tab_acc))
        tab_acc_test.append(1-np.mean(tab_acc))   
        resulat=s.run(scoo, feed_dict={ph_images: mnist_test_images[0:taille_batch]})

plot.ylim(0, 1)
plot.grid()
plot.plot(tab_acc_train, label="Train error")
plot.plot(tab_acc_test, label="Test error")
plot.legend(loc="upper right")
plot.show()

np.set_printoptions(formatter={'float': '{:0.3f}'.format})
for image in range(taille_batch):
    print("image", image)
    print("sortie du réseau:", resulat[image], np.argmax(resulat[image]))
    print("sortie attendue :", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
    cv2.imshow('image', mnist_test_images[image].reshape(28, 28))
    if cv2.waitKey()&0xFF==ord('q'):
        break






