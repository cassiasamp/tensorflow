import tensorflow as tf 

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)

sessao = tf.Session()

print(sessao.run(y, {a:3, b:4}))