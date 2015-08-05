#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 317
#define bits_in_byte 8

#define MAX_ATTRIBUTE_COUNT (16384 - 2) //-1 for write buffer, and -1 for other data (such as parameters)
#define MAX_INSTANCE_COUNT 16384
#define MAX_DATASET_SIZE 268435456

#ifdef __cplusplus
extern "C" {
#endif

	extern float aux_buffer[MAX_DATASET_SIZE];

	//---------//
	//--types--//
	//---------//

	typedef float Qfloat;
	typedef signed char schar;

	enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
	enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

	extern int libsvm_version;

	typedef struct {
		int lower;
		int upper;
	} limit_t;

	//
	// decision_function
	//
	struct decision_function {
		double *alpha;
		double rho;
	};

	//point struct
	typedef struct {
		int x;
		int y;
	} point_t;

	typedef struct {
		int index; //índice do atributo
		double value; //valor do atributo
		//double class_value;
	} svm_node;

	struct svm_problem {
		int count_instance; //number of instances in dataset
		int count_attribute; //number of attributes in dataset
		
		//TODO remove!
		double *class_value; //class value

		int *indexes; //order of dot products

		int elements; //attributes with value different from zero times (count_instance + 1)
		svm_node **node; //vector of instances, each instance consisting of a vector of svm_node; has overall size of elements * sizeof(svm_node)
	};

	struct svm_parameter {
		int svm_type;
		int kernel_type;
		int degree;	/* for poly */
		double gamma;	/* for poly/rbf/sigmoid */
		double coef0;	/* for poly/sigmoid */

		/* these are for training only */
		double cache_size; /* in MB */
		double eps;	/* stopping criteria */
		double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
		int nr_weight;		/* for C_SVC */
		int *weight_label;	/* for C_SVC */
		double* weight;		/* for C_SVC */
		double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
		double p;	/* for EPSILON_SVR */
		int shrinking;	/* use the shrinking heuristics */
		int probability; /* do probability estimates */
	};

	//
	// svm_model
	// 
	struct svm_model {
		struct svm_parameter param;	/* parameter */
		int nr_class;		/* number of classes, = 2 in regression/one class svm */
		int count_instance;			/* total #SV */
		svm_node **SV;		/* SVs (SV[l]) */
		double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
		double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
		double *probA;		/* pariwise probability information */
		double *probB;
		int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

		/* for classification only */

		int *label;		/* label of each class (label[k]) */
		int *nSV;		/* number of SVs for each class (nSV[k]) */
		/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
		/* XXX */
		int free_sv;		/* 1 if svm_model is created by svm_load_model*/
		/* 0 if svm_model is created by svm_train */
	};

	//-------------//
	//--functions--//
	//-------------//

	//functions
	//gets the position at dotProductValue matrix where the dot product of instance0 and instance1 is located.
	extern inline int getDotProductIndex(int instance0, int instance1);
	extern void setGPUSocketPointer(void *pointer);
	extern bool isInstanceEqual(int gpu_index, const svm_node *local, int count_instance, int count_attribute);
	extern void printInstances(int gpu_index, const svm_node *local, int count_instance, int count_attribute);
	extern int testAll(int *indexes, svm_node **node, int count_instance, int count_attribute);

	extern limit_t *write;
	extern limit_t *head;

	struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
	void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

	int svm_save_model(const char *model_file_name, const struct svm_model *model);
	struct svm_model *svm_load_model(const char *model_file_name);

	int svm_get_svm_type(const struct svm_model *model);
	int svm_get_nr_class(const struct svm_model *model);
	void svm_get_labels(const struct svm_model *model, int *label);
	void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
	int svm_get_nr_sv(const struct svm_model *model);
	double svm_get_svr_probability(const struct svm_model *model);

	double svm_predict_values(const struct svm_model *model, const svm_node *x, double* dec_values);
	double svm_predict(const struct svm_model *model, const svm_node *x);
	double svm_predict_probability(const struct svm_model *model, const svm_node *x, double* prob_estimates);

	void svm_free_model_content(struct svm_model *model_ptr);
	void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
	void svm_destroy_param(struct svm_parameter *param);

	const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
	int svm_check_probability_model(const struct svm_model *model);

	void svm_set_print_string_function(void (*print_func)(const char *));
	
	//-----------//
	//--classes--//
	//-----------//

	//
	// Kernel Cache
	//
	// count_instance is the number of total data items
	// size is the cache size limit in bytes
	//
	class Cache {
	public:
		Cache(int count_instance, long int size);
		~Cache();

		// request data [0,len)
		// return some position p where [p,len) need to be filled
		// (p >= len if nothing needs to be filled)
		int get_data(const int index, Qfloat **data, int len);
		void swap_index(int i, int j);
	private:
		int count_instance;
		long int size;
		struct head_t {
			head_t *prev, *next;	// a circular list
			Qfloat *data;
			int len;		// data[0,len) is cached in this entry
		};

		head_t *head;
		head_t lru_head;
		void lru_delete(head_t *h);
		void lru_insert(head_t *h);
	};

	//
	// Kernel evaluation
	//
	// the static method k_function is for doing single kernel evaluation
	// the constructor of Kernel prepares to calculate the count_instance*count_instance kernel matrix
	// the member function get_Q is for getting one column from the Q Matrix
	//
	class QMatrix {
	public:
		virtual Qfloat *get_Q(int column, int len) const = 0;
		virtual double *get_QD() const = 0;
		virtual void swap_index(int i, int j) const = 0;
		virtual ~QMatrix() {}
	};

	class Kernel : public QMatrix {
	public:
		const static int _KERNEL_LINEAR = 1;
		const static int _KERNEL_POLY = 2;
		const static int _KERNEL_RBF = 4;
		const static int _KERNEL_SIGMOID = 8;
		const static int _KERNEL_PRECOMPUTED = 16;
		
		Kernel(svm_node *const *node, int count_instance, int count_attribute, int elements, int *indexes, const svm_parameter& param);
		virtual ~Kernel();

		static double k_function(const svm_node *node, const svm_node *y, const svm_parameter& param);
		virtual Qfloat *get_Q(int column, int len) const = 0;
		virtual double *get_QD() const = 0;
		virtual void swap_index(int i, int j) const;

		static double dot(const svm_node *px, const svm_node *py);
		static float dot_float(const svm_node *px, const svm_node *py);

	protected:

		const svm_node **node;
		int count_attribute;
		int count_instance;
		double *x_square; 

		int elements;
		int *indexes;

		int kernel_mode;

		double (Kernel::*kernel_function)(int i, int j) const;

		// svm_parameter
		const int kernel_type;
		const int degree;
		const double gamma;
		const double coef0;

	private:
		double kernel_linear(int i, int j) const;
		double kernel_poly(int i, int j) const;
		double kernel_rbf(int i, int j) const;
		double kernel_sigmoid(int i, int j) const;
		double kernel_precomputed(int i, int j) const;
	};

	// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
	// Solves:
	//
	//	min_psvm 0.5(\alpha^T Q \alpha) + p^T \alpha
	//
	//		y^T \alpha = \delta
	//		y_i = +1 or -1
	//		0 <= alpha_i <= Cp for y_i = 1
	//		0 <= alpha_i <= Cn for y_i = -1
	//
	// Given:
	//
	//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
	//	count_instance is the size of vectors and matrices
	//	eps is the stopping tolerance
	//
	// solution will be put in \alpha, objective value will be put in obj
	//
	class Solver {
	public:
		Solver() {};
		virtual ~Solver() {};

		struct SolutionInfo {
			double obj;
			double rho;
			double upper_bound_p;
			double upper_bound_n;
			double r;	// for Solver_NU
		};

		void Solve(int count_instance, const QMatrix& Q, const double *p_, const schar *y_,
			double *alpha_, double Cp, double Cn, double eps,
			SolutionInfo* si, int shrinking);

	protected:
		int active_size;
		schar *y;
		double *G;		// gradient of objective function
		enum { LOWER_BOUND, UPPER_BOUND, FREE };
		char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
		double *alpha;
		const QMatrix *Q;
		const double *QD;
		double eps;
		double Cp, Cn;
		double *p;
		int *active_set;
		double *G_bar;		// gradient, if we treat free variables as 0
		int count_instance;
		bool unshrink;	// XXX

		double get_C(int i);
		void update_alpha_status(int i);
		bool is_upper_bound(int i);
		bool is_lower_bound(int i);
		bool is_free(int i);

		void swap_index(int i, int j);
		void reconstruct_gradient();
		virtual int select_working_set(int &i, int &j);
		virtual double calculate_rho();
		virtual void do_shrinking();

	private:
		bool be_shrunk(int i, double Gmax1, double Gmax2);
	};

	//
	// Solver for nu-svm classification and regression
	//
	// additional constraint: e^T \alpha = constant
	//
	class Solver_NU : public Solver {
	public:
		Solver_NU();
		void Solve(int count_instance, const QMatrix& Q, const double *p, const schar *y, double *alpha, double Cp, double Cn, double eps, SolutionInfo* si, int shrinking);

	private:

		SolutionInfo *si;
		int select_working_set(int &i, int &j);
		double calculate_rho();
		bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
		void do_shrinking();
	};

	//
	// Q matrices for various formulations
	//
	class SVC_Q : public Kernel {
	public:

		SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_);// : Kernel(prob.node, prob.count_instance, prob.count_attribute, prob.elements, param);

		Qfloat *get_Q(int i, int len) const;

		double *get_QD() const;
		void swap_index(int i, int j) const;

		~SVC_Q();

	private:
		schar *y;
		Cache *cache;
		double *QD;
	};

	class ONE_CLASS_Q : public Kernel {
	public:
		ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param);

		Qfloat *get_Q(int i, int len) const;
		double *get_QD() const;
		void swap_index(int i, int j) const;
		~ONE_CLASS_Q();
	private:
		Cache *cache;
		double *QD;
	};

	class SVR_Q : public Kernel {
	public:

		SVR_Q(const svm_problem& prob, const svm_parameter& param);
		void swap_index(int i, int j) const;
		Qfloat *get_Q(int i, int len) const;
		double *get_QD() const;
		~SVR_Q();

	private:
		int count_instance;
		Cache *cache;
		schar *sign;
		int *index;
		mutable int next_buffer;
		Qfloat *buffer[2];
		double *QD;
	};

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
