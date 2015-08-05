#include <CL/cl.h>

//in and out
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <ctype.h>
#include <errno.h>

#include <thread>

#include "psvm.h"
#include "GPUSocket.h"
#include "..\..\ClockUtility\ClockUtility.h"

using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

GPUSocket *_gpusocket = NULL;
thread *starter = NULL;

void startGPUSocket() {
	if(_gpusocket == NULL) {
		_gpusocket = new GPUSocket("HelloWorld_Kernel.cl", "multiply_matrices");
	}
}

void freeGPUSocket() {
	if(_gpusocket != NULL) {
		delete _gpusocket;
	}
}

void print_null(const char *s) {
}

void exit_with_help() {
	printf(
		"Usage: svm-train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
		"	1 -- nu-SVC		(multi-class classification)\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR	(regression)\n"
		"	4 -- nu-SVR		(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n"
		);
	std::exit(1);
}

void exit_input_error(int line_num) {
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	std::exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
svm_node *x_space;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input) {
	int len;

	if (fgets(line, max_line_len, input) == NULL) {
		return NULL;
	}

	while (strrchr(line, '\n') == NULL) {
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL) {
			break;
		}
	}
	return line;
}

void do_cross_validation() {
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.count_instance);

	svm_cross_validation(&prob, &param, nr_fold, target);
	if (param.svm_type == EPSILON_SVR ||
		param.svm_type == NU_SVR) {
		for (i = 0; i < prob.count_instance; i++) {
			double y = prob.class_value[i];
			double v = target[i];
			total_error += (v - y)*(v - y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n", total_error / prob.count_instance);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.count_instance*sumvy - sumv*sumy)*(prob.count_instance*sumvy - sumv*sumy)) /
			((prob.count_instance*sumvv - sumv*sumv)*(prob.count_instance*sumyy - sumy*sumy))
			);
	} else {
		for (i = 0; i < prob.count_instance; i++)
		if (target[i] == prob.class_value[i])
			++total_correct;
		printf("Cross Validation Accuracy = %g%%\n", 100.0*total_correct / prob.count_instance);
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name) {
	int i;
	void(*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	// parse options
	for (i = 1; i < argc; i++) {
		if (argv[i][0] != '-') break;
		if (++i >= argc)
			exit_with_help();
		switch (argv[i - 1][1]) {
			case 's':
			param.svm_type = atoi(argv[i]);
			break;
			case 't':
			param.kernel_type = atoi(argv[i]);
			break;
			case 'd':
			param.degree = atoi(argv[i]);
			break;
			case 'g':
			param.gamma = atof(argv[i]);
			break;
			case 'r':
			param.coef0 = atof(argv[i]);
			break;
			case 'n':
			param.nu = atof(argv[i]);
			break;
			case 'm':
			param.cache_size = atof(argv[i]);
			break;
			case 'c':
			param.C = atof(argv[i]);
			break;
			case 'e':
			param.eps = atof(argv[i]);
			break;
			case 'p':
			param.p = atof(argv[i]);
			break;
			case 'h':
			param.shrinking = atoi(argv[i]);
			break;
			case 'b':
			param.probability = atoi(argv[i]);
			break;
			case 'q':
			print_func = &print_null;
			i--;
			break;
			case 'v':
			cross_validation = 1;
			nr_fold = atoi(argv[i]);
			if (nr_fold < 2) {
				fprintf(stderr, "n-fold cross validation: n must >= 2\n");
				exit_with_help();
			}
			break;
			case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
			param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
			param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
			param.weight[param.nr_weight - 1] = atof(argv[i]);
			break;
			default:
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
			exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if (i >= argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if (i < argc - 1)
		strcpy(model_file_name, argv[i + 1]);
	else {
		char *p = strrchr(argv[i], '/');
		if (p == NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name, "%s.model", p);
	}
}

void inline previewDataset(FILE *fp, svm_problem *prob, int *width_values) {
	prob->elements = 0;
	prob->count_instance = 0;
	prob->count_attribute = 0;
	
	while (readline(fp) != NULL) {
		char *last, *p = strtok(line, " \t"); // label

		// features
		while (true) {
			last = p;
			p = strtok(NULL, " \t");
			
			if (p == NULL || *p == '\n') { // check '\n' as ' ' may be after the last feature
				int index = atoi(strtok(last, ":"));

				if(index > prob->count_attribute) {
					prob->count_attribute = index;
				}
				
				break;
			}
			prob->elements += 1;
		}
		prob->elements += 1;
		prob->count_instance += 1;
	}

	*width_values = (prob->count_attribute + (4 - (prob->count_attribute % 4)));

	rewind(fp);
}

void inline processDataset(FILE *fp, svm_problem *prob, int width_values) {
	prob->class_value = Malloc(double, prob->count_instance);
	prob->indexes = Malloc(int, prob->count_instance);
	prob->node = Malloc(svm_node *, prob->count_instance); 
	x_space = Malloc(svm_node, prob->elements); 

	starter->join();	
	setGPUSocketPointer(_gpusocket); 
	_gpusocket->createImage(prob->count_attribute, prob->count_instance);
	
	cl_ulong 
		avaiable = _gpusocket->getAvaiableGlobalMemory(),
		required = prob->count_attribute * prob->count_instance * sizeof(float);
	
	int j = 0, last_index, optimal = 0;
	char *endptr, *idx, *val, *label;
	
	for(int n = 0; ((n * prob->count_attribute * sizeof(float)) + (n * sizeof(float)) + 1) < avaiable; n++) {
		optimal++;
	}

	for (int i = 0; i < prob->count_instance; i++) {
		readline(fp);

		last_index = -1; //strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		prob->indexes[i] = i; //for now, index is the same as regular dataset
		prob->node[i] = &x_space[j]; //points to pre-allocated space

		label = strtok(line, " \t\n");
		if (label == NULL) { // empty line
			exit_input_error(i + 1);
		}

		prob->class_value[i] = strtod(label, &endptr); //class value
		if (endptr == label || *endptr != '\0') {
			exit_input_error(i + 1);
		}

		int current_index = 0;

		//cleans aux_buffer values
		for(int z = 0; z < prob->count_attribute + 1; z++) {
			aux_buffer[z] = 0.f;
		}

		while (true) { //iterates through attributes of instance
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL) {
				break;
			}

			//writes index value
			errno = 0;
			current_index = (int)strtol(idx, &endptr, 10);
			x_space[j].index = current_index; 

			if (endptr == idx || errno != 0 || *endptr != '\0' || current_index <= last_index) {
				exit_input_error(i + 1);
			} else {
				last_index = current_index; 
			}

			//writes attribute value
			errno = 0;

			//write value to aux_buffer
			aux_buffer[current_index-1] = (float)strtod(val, &endptr);
			x_space[j].value = aux_buffer[current_index-1]; 

			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
				exit_input_error(i + 1);
			} 

			j++;
			//current_index++; //for null value
		} //end while true
		
		if(param.kernel_type == PRECOMPUTED) {
			int value = (int)aux_buffer[0]; 
			if(value <= 0 || value > prob->count_attribute) {
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				std::exit(1);
			}
		}

		x_space[j].index = -1; //break data index 
		x_space[j].value = 0.f; //break data value
		j++;

		//writes one register to the gpu
		_gpusocket->writeInstanceToImage(&aux_buffer[0], width_values, i);

	} //end for i + n < prob->count_instance

	if (param.gamma == 0 && prob->count_attribute > 0) {
		param.gamma = 1.0 / prob->count_attribute;
	}

	///*
	if (param.kernel_type == PRECOMPUTED) {
		for (int i = 0; i<prob->count_instance; i++) {
			if(prob->node[i][0].index != 0) {
				fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
				std::exit(1);
			}
			if((int)prob->node[i][0].value <= 0 || (int)prob->node[i][0].value > prob->count_attribute) {
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				std::exit(1);
			}
		}
	}
	//*/
}

// read in a problem (in svmlight format)
void read_problem(const char *filename) {
	FILE *fp = fopen(filename, "r");
	
	if (fp == NULL) {
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	int width_values;

	//reads the problem once, counting the number of attributes and instances
	previewDataset(fp, &prob, &width_values);

	if((prob.count_attribute * prob.count_instance) > MAX_DATASET_SIZE) {
		fprintf(
			stderr, 
			"Dataset is bigger than GPU can support. Current dataset size (bytes): %d\n Avaiable space (bytes): %d\n",
			prob.count_attribute * prob.count_instance * sizeof(float),
			MAX_DATASET_SIZE * sizeof(float)
		);
		exit(1);
	}

	//read problem
	processDataset(fp, &prob, width_values);
	
	fclose(fp);
	printTimeElapsed("Done reading dataset at ", "\n");
}

int train(int argc, char **argv) {
	starter = new thread(startGPUSocket);

	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob, &param);

	if (error_msg) {
		fprintf(stderr, "ERROR: %s\n", error_msg);
		std::exit(1);
	}

	if (cross_validation) {
		do_cross_validation();
	} else {
		model = svm_train(&prob, &param);
		if (svm_save_model(model_file_name, model)) {
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			std::exit(1);
		}
		svm_free_and_destroy_model(&model);
	}
	svm_destroy_param(&param);
	std::free(prob.class_value);
	std::free(prob.node); 
	std::free(prob.indexes);
	std::free(x_space); 
	std::free(line);
	freeGPUSocket();

	return 0;
}

int main(int argc, char **argv) { 
	startClock();
	int response = train(argc, argv);
	printTimeElapsed("Total time elapsed: ", "\n");
	return response;
}