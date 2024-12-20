
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import multiprocessing

class Input:
    #Class variables (remember that they are different than instance variables, and all instances or objects have access to them)
    path = '' #The location where the results will be written
    mode = ''   #Type of prediction task that the object will be used for, i.e., "event_prediction", "timestamp_prediction", "event_timestamp_prediction"
    dataset_name = '' #Name of the input dataset
    prefix_len = ''  #It is a number that shows the length of the considered prefixes
    batch = ''       #It is a number that shows size of used batch
    design_matrix = ''  # A matrix that stores the designed matrix (each activity is shown by one hot vector)
    design_matrix_padded = '' #A design matrix that is padded after creating the prefixes
    y = '' #The ground truth labels related to the "design_matrix_padded"
    unique_event = ''  #The list of unique events, including end of trace as "0"
    selected_columns = '' # List of considered columns, including event and other information
    timestamp_loc = ''    # The column index for timestamp feature
    train_inds=''     #Index of training instances
    test_inds=''      #Index of test instances
    validation_inds=''     #Index of validation instances
    train_loader = ''
    test_loader = ''
    validation_loader = ''




    #class methods can be called without creating objects (they have cls instead of self)
    #start from here
    @classmethod
    def run(cls, path, prefix, batch_size, mode="event_prediction"):
        '''
        This method is the starting point for preparing an object to be used later in different prediction tasks.

        @param path: The location of the event log (CSV or Pickle)
        @param prefix: Size of the prefix
        @param batch_size: Size of batch
        @param mode: "event_prediction", "timestamp_prediction", "event_timestamp_prediction"
        @return: None (sets class attributes)
        '''
        cls.prefix_len = prefix
        cls.batch = batch_size
        cls.dataset_name = os.path.splitext(os.path.basename(path))[0]  # Extracts dataset name without extension
        cls.mode = mode
        cls.path = os.path.join(os.getcwd(), cls.dataset_name, mode, f'prefix_{cls.prefix_len}')

        # Check file extension and read the file accordingly
        file_extension = path.split('.')[-1].lower()

        if file_extension == 'csv':
            data_augment = cls.__read_csv_massive(path)
        elif file_extension == 'pkl':
            try:
                with open(path, "rb") as f:
                    data_augment = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Error reading pickle file: {e}")
            print("The head of augmented data (with remaining and duration times):\n", data_augment.head(10))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are CSV and Pickle.")

        # Check if data_augment is a valid DataFrame
        if data_augment is None or not isinstance(data_augment, pd.DataFrame):
            raise ValueError("Data augmentation resulted in an invalid DataFrame. Please check the data source.")

        # Creating a design matrix that shows one-hot vector representation for activity IDs
        cls.design_matrix = cls.__design_matrix_creation(data_augment)

        #Creating prefix
        cls.prefix_creating(prefix, mode)

        #Determining the train,test, and validation sets
        cls.train_valid_test_index()

        # #Correcting test data
        # cls.testData_correction()

        #Creating minibatch
        cls.mini_batch_creation(batch_size)



    #################################################################################
    #Reading the CSV file
    @classmethod
    def __read_csv(cls, path):
        '''
        The input CSV is a file where the events are encoded into numerical activity IDs.
        '''
        # Reading CSV file
        dat = pd.read_csv(path)
        print("Types before:", dat.dtypes)

        # Changing the data type from integer to category
        dat['ActivityID'] = dat['ActivityID'].astype('category')
        dat['CompleteTimestamp'] = pd.to_datetime(dat['CompleteTimestamp'])

        print("Types after:", dat.dtypes)
        print("Columns:", dat.columns)

        # Grouping the data by 'CaseID'
        dat_group = dat.groupby('CaseID')
        print("Original data (first 5 rows):\n", dat.head())

        # Data Preparation
        data_augment = []  # List to hold data, to avoid inefficient append in loop
        total_iter = len(dat_group)  # Total number of groups (CaseID)
        pbar = tqdm(total=total_iter, desc="Processing cases")

        # Iterating over groups in the DataFrame
        for name, gr in dat_group:
            # Sorting by time within each group (CaseID)
            gr = gr.sort_values(by='CompleteTimestamp')  # Sorting in-place

            # Computing the duration time (difference between timestamps)
            duration_time = gr['CompleteTimestamp'].diff() / np.timedelta64(1, 'D')  # In days
            duration_time.iloc[0] = 0  # Fill NaN with 0 for the first row

            # Computing the remaining time (time to finish)
            length = len(duration_time)
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(length)]

            # Adding duration and remaining time to the group DataFrame
            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time

            # Append the processed group to the list
            data_augment.append(gr)

            # Update progress bar
            pbar.update(1)

        pbar.close()

        # Concatenate all groups into a single DataFrame
        data_augment = pd.concat(data_augment, ignore_index=True)
        print("Dataset with duration and remaining times:\n", data_augment.head(10))

        # Save the augmented data to a pickle file for future use
        name = path.split(".")[0].split("/")[-1]  # Get the base file name (without extension)
        pickle_file = f"{name}_augmented.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(data_augment, f)

        print(f"Data has been saved to {pickle_file}")

        return data_augment
    ################################################################################
    # Reading the CSV file
    @classmethod
    def __read_csv_massive(cls, path):
        '''
        The input CSV is a file where the events are encoded into numerical activity IDs
        '''
        # Reading the CSV file
        try:
            dat = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading the CSV file: {e}")

        print("Types before:", dat.dtypes)

        # Ensure columns exist before proceeding
        if 'ActivityID' not in dat.columns or 'CompleteTimestamp' not in dat.columns:
            raise ValueError("CSV file must contain 'ActivityID' and 'CompleteTimestamp' columns.")

        # Changing the data type from integer to category for ActivityID and datetime for CompleteTimestamp
        dat['ActivityID'] = dat['ActivityID'].astype('category')
        dat['CompleteTimestamp'] = pd.to_datetime(dat['CompleteTimestamp'])

        print("Types after:", dat.dtypes)
        print("Columns:", dat.columns)

        # Group by 'CaseID'
        dat_group = dat.groupby('CaseID')
        print("Original data (first 5 rows):\n", dat.head())
        print("Group by data (first 5 groups):\n", list(dat_group)[:5])

        # Create as many processes as there are CPUs on your machine
        num_processes = multiprocessing.cpu_count()

        # Calculate the chunk size based on the number of rows and processes
        chunk_size = int(dat.shape[0] / num_processes)

        # Handle case where the data isn't perfectly divisible by num_processes
        chunks = [dat.iloc[dat.index[i:i + chunk_size]] for i in range(0, dat.shape[0], chunk_size)]

        # Create a multiprocessing pool with `num_processes` processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Apply the function to each chunk in the list
            results = pool.map(cls.func, chunks)

        # Concatenate the list of DataFrames into a single DataFrame
        results = pd.concat(results, ignore_index=True)

        # Save the results to a pickle file
        name = path.split(".")[0].split("/")[-1]  # Get the base name of the file (without extension)
        pickle_file = f"{name}_augmented.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(results, f)

        print(f"Data has been saved to {pickle_file}")
        return results

    ######################################################################################
    @classmethod
    def func(cls, dat):
        # Data Preparation (used by read_csv_massive)
        # Iterating over groups in Pandas DataFrame
        data_augment = []  # Use a list to collect DataFrames
        dat_group = dat.groupby('CaseID')

        total_iter = len(dat_group.ngroup())
        pbar = tqdm(total=total_iter)
        for name, gr in dat_group:
            # Sorting by time
            gr.sort_values(by=['CompleteTimestamp'], inplace=True)

            # Computing the duration time in seconds by differencing x[t+1]-x[t]
            duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 'D')

            # Filling NaN with 0
            duration_time.iloc[0] = 0

            # Computing the remaining time
            length = duration_time.shape[0]
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(length)]

            # Adding computed columns to the DataFrame
            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time

            # Append the processed group to the list
            data_augment.append(gr)

            pbar.update(1)
        pbar.close()

        # Concatenate all processed groups into a single DataFrame
        data_augment = pd.concat(data_augment, ignore_index=True)

        # Return the augmented DataFrame
        return data_augment
    #######################################################################################
    #Creating a design matrix (one hot vector representation)
    @classmethod
    def __design_matrix_creation(cls, data_augment):
        '''
        data_augment is pandas dataframe created after reading CSV input by "read_csv()" method
        '''

        # check if data augment correctly passed
        print(type(data_augment))  # Should print <class 'pandas.core.frame.DataFrame'>
        print(data_augment.head())  # Preview the DataFrame to confirm its content
        if 'ActivityID' not in data_augment.columns:
            raise ValueError("The DataFrame does not contain the 'ActivityID' column.")

        # Creating a desing matrix (one hot vectors for activities), End of line (case) is denoted by class 0
        unique_event = sorted(data_augment['ActivityID'].unique())
        cls.unique_event = [0] + unique_event
        print("uniqe events:", unique_event)

        l = []
        for index, row in tqdm(data_augment.iterrows()):
            temp = dict()
            '''
            temp ={1: 0,
                  2: 0,
                  3: 1,
                  4: 0,
                  5: 0,
                  6: 0,
                  '0':0,
                  'duration_time': 0.0,
                  'remaining_time': 1032744.0}
            '''

            # Defning the columns we consider
            keys = ['0'] + list(unique_event) + ['duration_time', 'remaining_time']
            for k in keys:
                if (k == row['ActivityID']):
                    temp[k] = 1
                else:
                    temp[k] = 0

            temp['class'] = row['ActivityID']
            temp['duration_time'] = row['duration_time']
            temp['remaining_time'] = row['remaining_time']
            temp['CaseID'] = row['CaseID']

            l.append(temp)

        # Creating a dataframe for dictionary l
        design_matrix = pd.DataFrame(l)
        print("The design matrix is:\n", design_matrix.head(10))
        return design_matrix
    ################################################################################
    # Creating the desing matrix based on given prefix.
    @classmethod
    def prefix_creating(cls, prefix=2, mode = 'event_prediction'):


        if (mode == "timestamp_prediction"):
            clsN = cls.design_matrix.columns.get_loc('duration_time')
        elif (mode == "event_prediction"):
            clsN = cls.design_matrix.columns.get_loc('class')
        elif (mode == 'event_timestamp_prediction'):
            clsN = [cls.design_matrix.columns.get_loc('duration_time')] + [cls.design_matrix.columns.get_loc('class')]
            cls.timestamp_loc = cls.design_matrix.columns.get_loc('duration_time')
            cls.selected_columns = cls.unique_event + [cls.timestamp_loc]



        group = cls.design_matrix.groupby('CaseID')
        # Iterating over the groups to create tensors
        temp = []
        temp_shifted = []
        for name, gr in group:
            gr = gr.drop('CaseID', axis=1)
            # For each group, i.e., view, we create a new dataframe and reset the index
            gr = gr.copy(deep=True)
            gr = gr.reset_index(drop=True)

            # adding a new row at the bottom of each case to denote the end of a case
            new_row = [0] * gr.shape[1]
            gr.loc[gr.shape[0]] = new_row
            gr.iloc[gr.shape[0] - 1, gr.columns.get_loc('0')] = 1  # End of line is denoted by class 0

            gr_shift = gr.shift(periods=-1, fill_value=0)
            gr_shift.loc[gr.shape[0] - 1, '0'] = 1

            # Selecting only traces that has length greater than the defined prefix

            if (gr.shape[0] - 1 > prefix):
                for i in range(gr.shape[0]):
                    # if (i+prefix == gr.shape[0]):
                    #   break
                    # print(gr.iloc[i:i+prefix])
                    temp.append(torch.tensor(gr.iloc[i:i + prefix].values, dtype=torch.float, requires_grad=False))


                    # Storing the next element after the prefix as the prediction class
                    try:
                        # print("the prediction:", "the i", i ,gr.iloc[i+prefix,cls])
                        temp_shifted.append(
                            torch.tensor([gr.iloc[i + prefix, clsN]], dtype=torch.float, requires_grad=False))
                    except IndexError:
                        # Printing the end of sequence
                        # print("the prediction:", "ESLE the i", i ,0)
                        temp_shifted.append(torch.tensor([np.float16(0)], dtype=torch.float, requires_grad=False))
                    # print("****************************")


            # break
        desing_matrix_padded = pad_sequence(temp, batch_first=True)
        desing_matrix_shifted_padded = pad_sequence(temp_shifted, batch_first=True)

        # Saving the variables
        cls.design_matrix_padded = desing_matrix_padded
        cls.y = desing_matrix_shifted_padded
        #return desing_matrix_padded, desing_matrix_shifted_padded


        #Applying pad corrections
        cls.__pad_correction()

        print("The dimension of designed matrix:", cls.design_matrix_padded.size())
        print("The dim of ground truth:", cls.y.size())
        print("The prefix considered so far:", cls.design_matrix_padded.size()[1])


    ########################################################################################
    @classmethod
    def __pad_correction(cls):

        for i in range(cls.design_matrix_padded.size()[0]):
            u = (cls.design_matrix_padded[i, :, 0] == 1).nonzero()

            try:
                cls.design_matrix_padded[i, :, 0][u:] = 1
            except TypeError:
                pass
    ##################################################################################
    @classmethod
    def train_valid_test_index(cls):
        # Creating indexes by which obtaining train,test and validation sets


        train_inds = np.arange(0, round(cls.design_matrix_padded.size()[0] * .8))

        # Generating index for the test dataset
        test_inds = list(set(range(cls.design_matrix_padded.size()[0])).difference(set(train_inds)))
        validation_inds = test_inds[0:round(0.3 * len(test_inds))]
        test_inds = test_inds[round(0.3 * len(test_inds)):]


        cls.train_inds = train_inds
        cls.test_inds = test_inds
        cls.validation_inds = validation_inds
        print("Number of training instances:", len(train_inds))
        print("Number of testing instances:", len(test_inds))
        print("Number of validation instances:", len(validation_inds))

    #################################################################################
    @classmethod
    def testData_correction(cls):

        test_inds_new = []
        for i in cls.test_inds:
            # Checking how many stops are available in the prefix (we drop thoes prefixes with more than one stop element)
            # Remeber that the first column (index = 0) shows the end of sequence
            u = (cls.design_matrix_padded[i, :, 0] == 1).nonzero()
            if len(u) <= 1:
                test_inds_new.append(i)

        print("The number of test prefixes before correction:", len(cls.test_inds))
        print("The number of test prefixes after correction:", len(test_inds_new))

        cls.test_inds =  test_inds_new

    #################################################################################
    @classmethod
    def mini_batch_creation(cls, batch=4 ):
        train_data = TensorDataset(cls.design_matrix_padded[cls.train_inds], cls.y[cls.train_inds])
        train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)


        test_data = TensorDataset(cls.design_matrix_padded[cls.test_inds], cls.y[cls.test_inds])
        test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)

        validation_data = TensorDataset(cls.design_matrix_padded[cls.validation_inds], cls.y[cls.validation_inds])
        validation_loader = DataLoader(dataset=validation_data, batch_size=batch, shuffle=True)




        cls.train_loader = train_loader
        cls.test_loader = test_loader
        cls.validation_loader = validation_loader
        print("The minibatch is created!")





