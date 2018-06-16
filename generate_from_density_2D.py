'''Example code to generate samples for (given) target density function using NCE-GAN'''
import numpy as np
import os, time
from warnings import warn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow as tf

#Function to initiate weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def leaky_relu(x):
    return tf.nn.relu(x) - 0.2 * tf.nn.relu(-x)

#Heaviside step function for Tensorflow
def tf_step(x):
    return (1+tf.sign(x))/2

#Function to generate uniform samples
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

time_st = time.time()

ex_choice = 'Grid'
ex_list = ['Circle', 'Sine', 'Hill', 'Grid', 'Ring']
if not(ex_choice in ex_list):
    warn('ex_choice should be in the ex_list.')

G_loss_choice ='Minibatch'
G_loss_list = ['Vanilla', 'Minibatch', 'ln2']
if not(G_loss_choice in G_loss_list):
    warn('G_loss_choice should be in the G_loss_list.')

total_tr = 200000 #Total number of training iterations
epsilon = 1.0*10**(-6) #To make the target density is non-zero whenever the contrastive-noise distribution is non-zero
learning_rate = 5.0*10**(-4)
density_max_threshold = 1.0 #To penalize the case when estimated density value is too large

#Interval is chosen according to the ex_choice
if (ex_choice=='Circle')|(ex_choice=='Sine')|(ex_choice=='Hill'):
    Interval = np.array([[-1,1],[-1,1]])
elif (ex_choice=='Ring'):
    Interval = np.array([[-1.5,1.5],[-1.5,1.5]])
elif (ex_choice=='Grid'):
    Interval = np.array([[-5,5],[-5,5]])
V = (Interval[0][1] - Interval[0][0])*(Interval[1][1] - Interval[1][0]) #Area (Interval)
test_frq = 2000
lin_number = 200 #To visualize the estimated density function

h1_dim = 50 #Number of neurons (1st hidden layer)
h2_dim = 50 #Number of neurons (2nd hidden layer)
x_dim = 1
y_dim = 1
noise_dim = 30 #Dimension for the noise which will be fed to the generator
N_Test = 2000
mb_size = 64 #Batch size

Z_noise = tf.placeholder(tf.float32, shape=[None, noise_dim]) #Noise that will be fed to the generator
X = tf.placeholder(tf.float32, shape=[None, x_dim])
Y = tf.placeholder(tf.float32, shape=[None, y_dim])
C_noise = tf.placeholder(tf.float32, shape=[None, x_dim + y_dim]) #Contrastive noise

#Define positions and standard deviations of clusters for grid and ring examples
grid_centers = np.array(np.meshgrid(np.linspace(-4,4,5),np.linspace(-4,4,5))).reshape([2,5**2])
grid_centers = grid_centers.transpose()
ring_theta = np.linspace(0,7/4*np.pi,8).reshape([-1,1])
ring_centers = np.concatenate([np.cos(ring_theta),np.sin(ring_theta)],1)
if (ex_choice=='Ring'):
    gaussian_std = 0.1
elif (ex_choice=='Grid'):
    gaussian_std = 0.25

#Tensorflow version of the target density distribution
def tf_Goal_density(inputs,example='Circle'):
    x = tf.gather(inputs,indices=[0],axis=1)
    y = tf.gather(inputs, indices=[1], axis=1)
    if example == 'Circle':
        density = tf_step(1-x**2-y**2)/np.pi
    elif example == 'Sine':
        density = tf_step(tf.sin(np.pi*x)-y) / 2
    elif example == 'Hill':
        density = (x+y+2) / 8
    elif example == 'Grid':
        density = tf.zeros_like(x)
        for i in range(5**2):
            density = density + tf.exp(-((x-grid_centers[i,0])**2+(y-grid_centers[i,1])**2)/(2*gaussian_std**2))/(2*np.pi*gaussian_std**2 * 5**2)
    elif example == 'Ring':
        density = tf.zeros_like(x)
        for i in range(8):
            density = density + tf.exp(-((x-ring_centers[i,0])**2+(y-ring_centers[i,1])**2)/(2*gaussian_std**2))/(2*np.pi*gaussian_std**2 * 8)
    density = (density + epsilon)/(1 + epsilon*V) #To make the target density is non-zero whenever the contrastive-noise distribution is non-zero
    return density

#Numpy version of the target density distribution
def Goal_density(x, y, example='Circle'):
    if example == 'Circle':
        density = np.heaviside(1-x**2-y**2,0.5)/np.pi
    elif example == 'Sine':
        density = np.heaviside(np.sin(np.pi*x)-y,0.5) / 2
    elif example == 'Hill':
        density = (x+y+2) / 8
    elif example == 'Grid':
        density = np.zeros_like(x)
        for i in range(5 ** 2):
            density = density + np.exp(-((x - grid_centers[i, 0]) ** 2 + (y - grid_centers[i, 1]) ** 2) / (2 * gaussian_std ** 2)) / \
                                (2 * np.pi * gaussian_std ** 2 * 5 ** 2)
    elif example == 'Ring':
        density = np.zeros_like(x)
        for i in range(8):
            density = density + np.exp(-((x - ring_centers[i, 0]) ** 2 + (y - ring_centers[i, 1]) ** 2) / (2 * gaussian_std ** 2)) / \
                                (2 * np.pi * gaussian_std ** 2 * 8)
    density = (density + epsilon) / (1 + epsilon * V)
    return density

'''G: Generator'''
G_W1 = tf.Variable(xavier_init([noise_dim, h1_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h1_dim]))
G_W2 = tf.Variable(xavier_init([h1_dim, h2_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[h2_dim]))
G_W3_X = tf.Variable(xavier_init([h2_dim, x_dim]))
G_b3_X = tf.Variable(tf.ones(shape=[x_dim]))
G_W3_Y = tf.Variable(xavier_init([h2_dim, y_dim]))
G_b3_Y = tf.Variable(tf.ones(shape=[y_dim]))

theta_G = [G_W1, G_W2, G_W3_X, G_W3_Y, G_b1, G_b2, G_b3_X, G_b3_Y]

def G(z):
    G_h1 = leaky_relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3_X = tf.matmul(G_h2, G_W3_X) + G_b3_X
    G_h3_Y = tf.matmul(G_h2, G_W3_Y) + G_b3_Y
    G_X = ((Interval[0][1]-Interval[0][0]))/2 * (tf.nn.softsign(G_h3_X) + (Interval[0][0]+Interval[0][1])/(Interval[0][1]-Interval[0][0]))
    G_Y = ((Interval[1][1]-Interval[1][0]))/2 * (tf.nn.softsign(G_h3_Y) + (Interval[1][0]+Interval[1][1])/(Interval[1][1]-Interval[1][0]))
    return G_X, G_Y

G_X_sample, G_Y_sample = G(Z_noise)
G_sample = tf.concat(axis=1, values=[G_X_sample, G_Y_sample])

'''D: Discriminator'''
D_W1 = tf.Variable(xavier_init([x_dim + y_dim, h1_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h1_dim]))
D_W2 = tf.Variable(xavier_init([h1_dim, h2_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[h2_dim]))
D_W3_goal = tf.Variable(xavier_init([h2_dim, 1]))
D_b3_goal = tf.Variable(tf.ones(shape=[1]))
D_W3_G = tf.Variable(xavier_init([h2_dim, 1]))
D_b3_G = tf.Variable(tf.ones(shape=[1]))

theta_D = [D_W1, D_W2, D_W3_goal, D_W3_G, D_b1, D_b2, D_b3_goal, D_b3_G]

def D(inputs):
    D_h1 = leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3_goal = tf.matmul(D_h2, D_W3_goal) + D_b3_goal
    D_h3_G = tf.matmul(D_h2, D_W3_G) + D_b3_G
    D_h3_CN = tf.log(tf.ones_like(D_h3_G) / V)
    D_logit = tf.concat(axis=1, values=[D_h3_goal, D_h3_G, D_h3_CN])
    D_logit_G = tf.concat(axis=1, values=[D_h3_goal, D_h3_G])
    return D_logit, tf.exp(D_h3_goal), tf.exp(D_h3_G), D_logit_G

den_target = tf_Goal_density(C_noise,ex_choice)

D_logit_G, D_h3_goal_G, D_h3_G_G, D_logit_G_G = D(G_sample)
D_logit_CN, D_h3_goal_CN, D_h3_G_CN, _ = D(C_noise)

D_MB_logit_G = tf.reduce_sum(D_logit_G,axis=0)
D_MB_logit_CN = tf.reduce_sum(D_logit_CN,axis=0)

label_goal = tf.concat([np.ones([mb_size,1]),np.zeros([mb_size,2])],axis=1)
label_G = tf.concat([np.zeros([mb_size,1]),np.ones([mb_size,1]),np.zeros([mb_size,1])],axis=1)
label_CN = tf.concat([np.zeros([mb_size,2]),np.ones([mb_size,1])],axis=1)

label_goal_G = tf.concat([np.ones([mb_size,1]),np.zeros([mb_size,1])],axis=1)

label_MB_goal = tf.concat([np.ones([1,1]),np.zeros([1,2])],axis=1)
label_MB_G = tf.concat([np.zeros([1,1]),np.ones([1,1]),np.zeros([1,1])],axis=1)
label_MB_CN = tf.concat([np.zeros([1,2]),np.ones([1,1])],axis=1)

D_h3_goal_CN_ = D_h3_goal_CN/(tf.reduce_mean(D_h3_goal_CN)*V)
D_loss_goal = tf.reduce_mean(den_target * tf.log(den_target/D_h3_goal_CN_ + epsilon)) + tf.nn.relu(D_h3_goal_CN_- density_max_threshold) # KL-divergence between the target distribution and the estimated target distribution
D_loss_G = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_G, labels=label_G))
D_loss_CN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_CN, labels=label_CN))

D_loss = 1 * D_loss_goal + (D_loss_G + D_loss_CN)

if G_loss_choice=='Vanilla':
    G_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_G_G, labels=label_goal_G))
elif G_loss_choice=='Minibatch':
    G_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_MB_logit_G, labels=label_MB_goal)) #Minibatch-wise loss for the generator
elif G_loss_choice=='ln2':
    G_loss =(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_G_G, labels=label_goal_G))-np.log(2))**2

D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)

#Define location to save results
if G_loss_choice=='Vanilla':
    folder_name = 'out_generate_from_density_2D/'
elif G_loss_choice=='Minibatch':
    folder_name = 'out_generate_from_density_2D_minibatch/'
elif G_loss_choice=='ln2':
    folder_name = 'out_generate_from_density_2D_ln2/'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

#To visualize the estimated density function
x_lin = np.linspace(Interval[0][0], Interval[0][1], lin_number)
y_lin = np.linspace(Interval[1][0], Interval[1][1], lin_number)
X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
U_lin = np.concatenate((X_lin.reshape([-1,1]),Y_lin.reshape([-1,1])),axis=1)
Z_lin = Goal_density(X_lin.reshape([-1,1]), Y_lin.reshape([-1,1]),ex_choice)
Z_lin_ = Z_lin.reshape([lin_number,lin_number])

#Save contour plot of the target density function
fig = plt.figure(10)
plt.contourf(X_lin, Y_lin, Z_lin_, 150, cmap='jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Target density')
plt.xlim([Interval[0][0], Interval[0][1]])
plt.xlim([Interval[1][0], Interval[1][1]])
plt.savefig(folder_name + 'Target_contour_plot.png', bbox_inches='tight')
plt.close(fig)

kl_test = np.ones([total_tr,2]) * np.nan
if (Interval[0]!=Interval[1]).all():
    warn('Note that automated scaling of U_sample is going to work only if Interval[0]==Interval[1].')

#Training & test of the NCE-GAN model
for it in range(total_tr):
    Z_noise_sample = sample_Z(mb_size, noise_dim)
    U_sample = (Interval[0][1]-Interval[0][0])/2 * (sample_Z(mb_size, x_dim + y_dim)+(Interval[0][0]+Interval[0][1])/(Interval[0][1]-Interval[0][0]))

    for i in range(3):
        sess.run(D_solver, feed_dict={Z_noise: Z_noise_sample, C_noise: U_sample})
    sess.run(G_solver, feed_dict={Z_noise: Z_noise_sample, C_noise: U_sample})

    if it%test_frq==0:
        U_sample = (Interval[0][1] - Interval[0][0]) / 2 * \
                   (sample_Z(N_Test, x_dim + y_dim) + (Interval[0][0] + Interval[0][1]) / (Interval[0][1] - Interval[0][0]))
        den_goal_test, den_goal2_test, den_G_test = sess.run([den_target, D_h3_goal_CN, D_h3_G_CN], feed_dict={C_noise: U_sample})
        den_goal2_test = den_goal2_test / (np.mean(den_goal2_test) * V)
        den_G_test = den_G_test / (np.mean(den_G_test) * V)

        # Estimated KL-divergence between the target distribution and the estimated density of generated sample distribution
        kl_test[it,0] = np.mean(den_goal_test * (np.log(den_goal_test+epsilon) - np.log(den_G_test+epsilon))) * V
        # Estimated KL-divergence between the estimated target distribution and the estimated density of generated sample distribution
        kl_test[it, 1] = np.mean(den_goal2_test * (np.log(den_goal2_test + epsilon) - np.log(den_G_test + epsilon))) * V
    else:
        kl_test[it,0] = kl_test[int(test_frq * np.floor(it / test_frq)),0]
        kl_test[it, 1] = kl_test[int(test_frq * np.floor(it / test_frq)),1]

    #Save current results
    if it%10000==(10000-1):
        print('Iter: {}'.format(it + 1))
        Z_noise_sample = sample_Z(N_Test, noise_dim)
        U_sample = (Interval[0][1]-Interval[0][0])/2 * (sample_Z(N_Test, x_dim + y_dim)+(Interval[0][0]+Interval[0][1])/(Interval[0][1]-Interval[0][0]))

        G_samples = sess.run(G_sample, feed_dict={Z_noise: Z_noise_sample, C_noise: U_sample})

        Z_noise_sample_lin = sample_Z(lin_number**2, noise_dim)
        den_goal_test, den_goal2_test, den_G_test = sess.run([den_target, D_h3_goal_CN, D_h3_G_CN], feed_dict={Z_noise: Z_noise_sample_lin, C_noise: U_lin})

        print('Area (target):',(np.mean(den_goal2_test)*V))
        print('Area (G):', (np.mean(den_G_test) * V))
        den_goal_test = den_goal_test/ (np.mean(den_goal_test)*V)
        den_goal2_test = den_goal2_test / (np.mean(den_goal2_test) * V)
        den_G_test = den_G_test / (np.mean(den_G_test) * V)

        print('KL divergence:',kl_test[it],'\n')

        #Save the plot of the generated samples
        fig = plt.figure(1)
        plt.clf()
        if ex_choice=='Circle':
            plt.plot(x_lin,(1-(np.abs(x_lin)**2))**0.5, 'k--')
            plt.plot(x_lin, -(1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
        elif ex_choice == 'Sine':
            plt.plot(x_lin, np.sin(np.pi*x_lin), 'k--')
        elif ex_choice == 'Hill':
            plt.plot(x_lin, -x_lin, 'k--')
        xy = np.vstack([G_samples[:,0], G_samples[:,1]])
        den = gaussian_kde(xy)(xy) #Only for colorization of generated samples
        plt.scatter(G_samples[:,0], G_samples[:,1], s=0.5, c=den, cmap='jet', edgecolors='face')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Generated samples')
        plt.xlim([Interval[0][0], Interval[0][1]])
        plt.xlim([Interval[1][0], Interval[1][1]])
        plt.savefig(folder_name + 'Generated_samples_{}.png'.format(str(it + 1)), bbox_inches='tight')
        plt.pause(1.0)

        #Save the contour plot of the estimated target distribution
        fig = plt.figure(2)
        plt.clf()
        plt.contourf(X_lin, Y_lin, den_goal2_test.reshape([lin_number,lin_number]), 150, cmap='jet')
        if ex_choice == 'Circle':
            plt.plot(x_lin, (1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
            plt.plot(x_lin, -(1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
        elif ex_choice == 'Sine':
            plt.plot(x_lin, np.sin(np.pi*x_lin), 'k--')
        elif ex_choice == 'Hill':
            plt.plot(x_lin, -x_lin, 'k--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Density estimation (goal)')
        plt.xlim([Interval[0][0], Interval[0][1]])
        plt.xlim([Interval[1][0], Interval[1][1]])
        plt.savefig(folder_name + 'Goal_test_contour_plot_{}.png'.format(str(it + 1)), bbox_inches='tight')
        plt.pause(1.0)

        #Save the contour plot of the estimated density of generated distribution
        fig = plt.figure(3)
        plt.clf()
        plt.contourf(X_lin, Y_lin, den_G_test.reshape([lin_number, lin_number]), 150, cmap='jet')
        if ex_choice == 'Circle':
            plt.plot(x_lin, (1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
            plt.plot(x_lin, -(1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
        elif ex_choice == 'Sine':
            plt.plot(x_lin, np.sin(np.pi*x_lin), 'k--')
        elif ex_choice == 'Hill':
            plt.plot(x_lin, -x_lin, 'k--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Density estimation (generated)')
        plt.xlim([Interval[0][0], Interval[0][1]])
        plt.xlim([Interval[1][0], Interval[1][1]])
        plt.savefig(folder_name + 'G_test_contour_plot_{}.png'.format(str(it + 1)), bbox_inches='tight')
        plt.pause(1.0)

        #Visualize the changes of estimated KL-divergence
        fig = plt.figure(4)
        plt.clf()
        plt.semilogy(kl_test[:it, 0], c='k') # Estimated KL-divergence between the target distribution and the estimated density of generated sample distribution
        plt.semilogy(kl_test[:it, 1], c='b') # Estimated KL-divergence between the estimated target distribution and the estimated density of generated sample distribution
        plt.xlabel('Iteration')
        plt.title('KL divergence')
        plt.pause(1.0)



plt.close()
plt.ioff()
sess.close()

#Save the plot of the generated samples
fig = plt.figure(1)
if ex_choice=='Circle':
    plt.plot(x_lin,(1-(np.abs(x_lin)**2))**0.5, 'k--')
    plt.plot(x_lin, -(1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
elif ex_choice == 'Sine':
    plt.plot(x_lin, np.sin(np.pi * x_lin), 'k--')
elif ex_choice == 'Hill':
    plt.plot(x_lin, -x_lin, 'k--')
xy = np.vstack([G_samples[:,0], G_samples[:,1]])
den = gaussian_kde(xy)(xy)
plt.scatter(G_samples[:,0], G_samples[:,1], s=0.5, c=den, cmap='jet', edgecolors='face')
plt.xlim([Interval[0][0], Interval[0][1]])
plt.xlim([Interval[1][0], Interval[1][1]])
plt.savefig(folder_name + 'Generated_samples_{}.png'.format(str(it + 1)), bbox_inches='tight')
plt.close(fig)

#Save the contour plot of the estimated target distribution
fig = plt.figure(2)
plt.clf()
plt.contourf(X_lin, Y_lin, den_goal2_test.reshape([lin_number,lin_number]), 150, cmap='jet')
if ex_choice=='Circle':
    plt.plot(x_lin, (1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
    plt.plot(x_lin, -(1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
elif ex_choice == 'Sine':
    plt.plot(x_lin, np.sin(np.pi * x_lin), 'k--')
elif ex_choice == 'Hill':
    plt.plot(x_lin, -x_lin, 'k--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Density estimation (goal)')
plt.xlim([Interval[0][0], Interval[0][1]])
plt.xlim([Interval[1][0], Interval[1][1]])
plt.savefig(folder_name + 'Goal_test_contour_plot_{}.png'.format(str(it + 1)), bbox_inches='tight')
plt.close(fig)

#Save the contour plot of the estimated density of generated distribution
fig = plt.figure(3)
plt.clf()
plt.contourf(X_lin, Y_lin, den_G_test.reshape([lin_number, lin_number]), 150, cmap='jet')
if ex_choice=='Circle':
    plt.plot(x_lin, (1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
    plt.plot(x_lin, -(1 - (np.abs(x_lin) ** 2)) ** 0.5, 'k--')
elif ex_choice == 'Sine':
    plt.plot(x_lin, np.sin(np.pi * x_lin), 'k--')
elif ex_choice == 'Hill':
    plt.plot(x_lin, -x_lin, 'k--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Density estimation (generated)')
plt.xlim([Interval[0][0], Interval[0][1]])
plt.xlim([Interval[1][0], Interval[1][1]])
plt.savefig(folder_name + 'G_test_contour_plot_{}.png'.format(str(it + 1)), bbox_inches='tight')
plt.close(fig)

#Visualize the changes of estimated KL-divergence
fig = plt.figure(4)
plt.semilogy(kl_test[:it,0], c='k') # Estimated KL-divergence between the target distribution and the estimated density of generated sample distribution
plt.semilogy(kl_test[:it,1], c='b') # Estimated KL-divergence between the estimated target distribution and the estimated density of generated sample distribution
plt.xlabel('Iteration')
plt.title('KL divergence')
plt.savefig(folder_name + 'KL_divergence.png', bbox_inches='tight')
plt.close(fig)

time_ed = time.time()
print('Time spend: ',time_ed-time_st)


