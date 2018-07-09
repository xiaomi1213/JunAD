#This the entrance
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
# The parameters for loading dataset
flags.DEFINE_string('dataset_name','MNIST','Supported: MNIST, CIFAR-10, ImageNet.')

# The parameters for loading pre_trained model
flags.DEFINE_string('model_name','baseCNN', 'Supported: baseCNN, Inception v3, resnet50, vgg19, and so on.')

# The parameters for selecting examples
flags.DEFINE_boolean('select', True, 'Select the correctly classified examples for the experiments.')
flags.DEFINE_integer('nb_examples',2000, 'The number of examples selected for attack.')
flags.DEFINE_boolean('balance_sampling',False,'Select the same number of examples for each class.')
flags.DEFINE_boolean('test_mode',False,'Only select one sample for each class.')

# The parameters for attacks
flags.DEFINE_string('attacks','fgsm?eps=0.1;BIM?eps=0.18&eps_iter=0.02;','Attack name and parameters in URL style, separated by semicolon.')

# The parameters for robustness evaluation
flags.DEFINE_string('robustness',' ','Supported: FeatureSqueezing')

# The parameters for detection evaluation
flags.DEFINE_string('detection',' ','Supported: FeatureSqueezing')
flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train and test dataset.')

# The parameters for saving the results and displaying the info
flags.DEFINE_string('path','./result','The root path to store the result, defaulted root path.')
flags.DEFINE_string('result_folder','results','The output folder for results.')
flags.DEFINE_boolean('verbose',False,'Stdout level. The hidden content will be saved to log files anyway.')



def main(argv=None):

    # 01 load the dataset and pre_prepare the dataset
    dataset = load_datasets('mnist',)
    X_train,Y_train,X_test,Y_test = dataset()

    # 02 define the model calculasion graph



    x = tf.placeholder(tf.float32, [None, img_height, img_width, img_channal])
    y = tf.placeholder(tf.float32, [None, num_class])

    # set session and backend
    sess = tf.Session()
    backend = keras.backend

    # 02_1 load the pre_trained model
    model = load_model(model_name, dataset_name)

    # 02_2 model train
    if not pre_trained:
        model_train()

    # 02_3 model test
    model_test()


    # 03 select examples to generate adversarial examples
    X_test_selcted = select(X_test)
    path = path_generate.select_path_generate()
    store_examples(X_test_selcted,path)
    visualize(X_test_selcted, num_visualize)


    # 04 Attack
    #select one or several adversaries to generate adversarial examples

    # 04_1 generate adversarial examples
    X_test_selected = get_examples(X_test_selected_path)
    show_the_example_index()

    # 04_2 generate adversarial examples
    X_test_adv = generate(attacks, X_test_select)
    store_examples(X_test_adv, path)

    # 04_3 implementing the attack
    Y_pred_adv = model(X_test_adv)
    success_rate = calculate_success_rate(Y_pred_adv, Y_test)
    mean_percentage  = calculate_percentage(Y_pred_adv)
    path = generate_path()
    store_result(success_rate, path)

    # 05 evaluate_robustness
    if not evaluation:
        params={'model':FLAG.model,
            'dataset':FLAG.dataset,
            'method':FLAG.robust_method,
            }
        result = evaluate_robustness(**params)
        display(result)
        path = generate_path(FLAG.path)
        store_result(result,path)


    # 06 evaluate_detection
    if not detection:
        params = {'model'FLAG.model,
        'dataset':FLAG.dataset,
        'method':FLAG.detect_method}
        result = evaluate_detection(**params)
        display(result)
        path = generate_path(FLAG.path)
        store_result(result,path)



if __name__ == __main__:
    main()