#include "psvm.h"

#include <thread>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <sstream>
#include <iostream>

#include "GPUSocket.h"

int libsvm_version = LIBSVM_VERSION;

#ifndef min_psvm
template <class T> static inline T min_psvm(T node, T y) { 
	return (node<y) ? node : y; 
}
#endif

#ifndef max_psvm
template <class T> static inline T max_psvm(T node, T y) { 
	return (node>y) ? node : y; 
}
#endif

template <class T> static inline void swap_psvm(T& node, T& y) { 
	T t = node; node = y; y = t; 
}

template <class S, class T> static inline void clone_psvm(T*& dst, S* src, int n) {
	dst = new T[n];
	memcpy((void *)dst, (void *)src, sizeof(T)*n);
}

static inline double powi(double base, int times) {
	double tmp = base, ret = 1.0;

	for (int t = times; t > 0; t /= 2) {
		if (t % 2 == 1) ret *= tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s) {
	fputs(s, stdout);
	fflush(stdout);
}
static void(*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt, ...) {
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap, fmt);
	vsprintf_s(buf, fmt, ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//dotProductValue
float aux_buffer[MAX_DATASET_SIZE];

//wizard block
GPUSocket *gpusocket = NULL; 

void setGPUSocketPointer(void *pointer) {
	gpusocket = (GPUSocket*)pointer;
}

/*
void printInstances(int gpu_index, const svm_node *local, int count_instance, int count_attribute) {
	const svm_node *ahead = local;

	gpusocket->run(count_attribute, count_instance, &aux_buffer[0], gpu_index, GET_INSTANCE);
	
	for(int n = 0; n < count_attribute; n++) {
		if(ahead->index == n) {
			cout << n << " " << ahead->value << " " << aux_buffer[n] << endl;
			ahead++;
		} else {
			cout << n << " " << "0" << " " << aux_buffer[n] << endl;
		}
	}
	cout << "------";
}


int testAll(int *indexes, svm_node **node, int count_instance, int count_attribute) {
	int count_wrong = 0;
	for(int n = 0; n < count_instance; n++) {
		if(!isInstanceEqual(indexes[n], node[n], count_instance, count_attribute)) {
			count_wrong++;
		}
	}
	return count_wrong;
}

bool isInstanceEqual(int gpu_index, const svm_node *local, int count_instance, int count_attribute) {
	const svm_node *ahead = local;

	gpusocket->run(count_attribute, count_instance, &aux_buffer[0], gpu_index, GET_INSTANCE);

	for(int n = 0; n < count_attribute; n++) {
		if(ahead->index == n) {
			if(ahead->value != aux_buffer[n]) {
				return false;
			} else {
				ahead++;
			}
		}
	}
	return true;
}
*/

//Cache
Cache::Cache(int l_, long int size_) :count_instance(l_), size(size_) {
	head = (head_t *)calloc(count_instance, sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= count_instance * sizeof(head_t) / sizeof(Qfloat);
	size = max_psvm(size, 2 * (long int)count_instance);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache() {
	for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h) {
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h) {
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len) {
	head_t *h = &head[index];
	if (h->len) lru_delete(h);
	int more = len - h->len;

	if (more > 0) {
		// free old space
		while (size < more) {
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat)*len);
		size -= more;
		swap_psvm(h->len, len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j) {
	if (i == j) return;

	if (head[i].len) lru_delete(&head[i]);
	if (head[j].len) lru_delete(&head[j]);
	swap_psvm(head[i].data, head[j].data);
	swap_psvm(head[i].len, head[j].len);
	if (head[i].len) lru_insert(&head[i]);
	if (head[j].len) lru_insert(&head[j]);

	if (i > j) swap_psvm(i, j);
	for (head_t *h = lru_head.next; h != &lru_head; h = h->next) {
		if (h->len > i) {
			if (h->len > j)
				swap_psvm(h->data[i], h->data[j]);
			else {
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

void Kernel::swap_index(int i, int j) const	{ 
	swap_psvm(node[i], node[j]);
	swap_psvm(indexes[i], indexes[j]);
	if (x_square) { //x_square != NULL
		swap_psvm(x_square[i], x_square[j]);
	}
}

double Kernel::kernel_linear(int i, int j) const {
	return dot(node[i], node[j]);
}
double Kernel::kernel_poly(int i, int j) const {
	return powi(gamma*dot(node[i], node[j]) + coef0, degree);
}
double Kernel::kernel_rbf(int i, int j) const {
	return exp(-gamma*(x_square[i] + x_square[j] - 2 * dot(node[i], node[j])));
}
double Kernel::kernel_sigmoid(int i, int j) const {
	return tanh(gamma*dot(node[i], node[j]) + coef0);
}
double Kernel::kernel_precomputed(int i, int j) const {
	return node[i][(int)(node[j][0].value)].value;
}

Kernel::Kernel(svm_node * const * node, int count_instance, int count_attribute, int elements, int *indexes, const svm_parameter& param)
: kernel_type(param.kernel_type), degree(param.degree), gamma(param.gamma), coef0(param.coef0) {
	switch (kernel_type) {
		case LINEAR:
		kernel_mode = _KERNEL_LINEAR;
		kernel_function = &Kernel::kernel_linear;
		break;
		case POLY:
		kernel_mode = _KERNEL_POLY;
		kernel_function = &Kernel::kernel_poly;
		break;
		case RBF:
		kernel_mode = _KERNEL_RBF;
		kernel_function = &Kernel::kernel_rbf;
		break;
		case SIGMOID:
		kernel_mode = _KERNEL_SIGMOID;
		kernel_function = &Kernel::kernel_sigmoid;
		break;
		case PRECOMPUTED:
		kernel_mode = _KERNEL_PRECOMPUTED;
		kernel_function = &Kernel::kernel_precomputed;
		break;
	}

	clone_psvm(this->node, node, count_instance); //TODO commented
	clone_psvm(this->indexes, indexes, count_instance);

	this->elements = elements;
	this->count_instance = count_instance;
	this->count_attribute = count_attribute;

	x_square = NULL;

	if (kernel_type == RBF) {
				
		x_square = new double [count_instance];

		//*
		gpusocket->run(count_attribute, count_instance, &aux_buffer[0], NULL, DIAGONAL);
		for (int i = 0; i < count_instance; i++) { 
			x_square[i] = aux_buffer[indexes[i]];
		}
		
		//*/

		//original source code
		/*
		for (int i = 0; i < count_instance; i++) { 
			x_square[i] = dot(node[i], node[i]);
		}
		//*/
	} 
}

Kernel::~Kernel() {
	delete[] node;
	if(x_square != NULL) {
		delete[] x_square;
	}
}

float Kernel::dot_float(const svm_node *px, const svm_node *py) {
	float sum = 0;
	while (px->index != -1 && py->index != -1) {
		if (px->index == py->index) {
			sum += (float)px->value * (float)py->value;
			++px;
			++py;
		} else {
			if (px->index > py->index) {
				++py;
			}
			else {
				++px;
			}
		}
	}
	return sum;
}

double Kernel::dot(const svm_node *px, const svm_node *py) {
	double sum = 0;
	while (px->index != -1 && py->index != -1) {
		if (px->index == py->index) {
			sum += px->value * py->value;
			++px;
			++py;
		} else {
			if (px->index > py->index) {
				++py;
			}
			else {
				++px;
			}
		}
	}
	return sum;
}

double Kernel::k_function(const svm_node *node, const svm_node *y,
	const svm_parameter& param) {
	switch (param.kernel_type) {
		case LINEAR:
			return dot(node, y);
		case POLY:
			return powi(param.gamma*dot(node, y) + param.coef0, param.degree);
		case RBF:
		{
			double sum = 0;
			while (node->index != -1 && y->index != -1) {
				if (node->index == y->index) {
					double d = node->value - y->value;
					sum += d*d;
					++node;
					++y;
				} else {
					if (node->index > y->index) {
						sum += y->value * y->value;
						++y;
					} else {
						sum += node->value * node->value;
						++node;
					}
				}
			}

			while (node->index != -1) {
				sum += node->value * node->value;
				++node;
			}

			while (y->index != -1) {
				sum += y->value * y->value;
				++y;
			}

			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(node, y) + param.coef0);
		case PRECOMPUTED:  //node: test (validation), y: SV
			return node[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

double Solver::get_C(int i) {
		return (y[i] > 0) ? Cp : Cn;
}

void Solver::update_alpha_status(int i) {
	if (alpha[i] >= get_C(i))
		alpha_status[i] = UPPER_BOUND;
	else if (alpha[i] <= 0)
		alpha_status[i] = LOWER_BOUND;
	else alpha_status[i] = FREE;
}

bool Solver::is_upper_bound(int i) { 
	return alpha_status[i] == UPPER_BOUND; 
}

bool Solver::is_lower_bound(int i) { 
	return alpha_status[i] == LOWER_BOUND; 
}

bool Solver::is_free(int i) { 
	return alpha_status[i] == FREE; 
}

void Solver::swap_index(int i, int j) {
	Q->swap_index(i, j);
	swap_psvm(y[i], y[j]);
	swap_psvm(G[i], G[j]);
	swap_psvm(alpha_status[i], alpha_status[j]);
	swap_psvm(alpha[i], alpha[j]);
	swap_psvm(p[i], p[j]);
	swap_psvm(active_set[i], active_set[j]);
	swap_psvm(G_bar[i], G_bar[j]);
}

void Solver::reconstruct_gradient() {
	// reconstruct inactive elements of G from G_bar and free variables

	if (active_size == count_instance) return;

	int i, j;
	int nr_free = 0;

	for (j = active_size; j < count_instance; j++)
		G[j] = G_bar[j] + p[j];

	for (j = 0; j < active_size; j++)
	if (is_free(j))
		nr_free++;

	if (2 * nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*count_instance > 2 * active_size*(count_instance - active_size)) {
		for (i = active_size; i < count_instance; i++) {
			const Qfloat *Q_i = Q->get_Q(i, active_size);
			for (j = 0; j < active_size; j++)
			if (is_free(j))
				G[i] += alpha[j] * Q_i[j];
		}
	} else {
		for (i = 0; i < active_size; i++)
		if (is_free(i)) {
			const Qfloat *Q_i = Q->get_Q(i, count_instance);
			double alpha_i = alpha[i];
			for (j = active_size; j < count_instance; j++)
				G[j] += alpha_i * Q_i[j];
		}
	}
}

void Solver::Solve(int count_instance, const QMatrix& Q, const double *p_, const schar *y_,
	double *alpha_, double Cp, double Cn, double eps,
	SolutionInfo* si, int shrinking) {
	this->count_instance = count_instance;
	this->Q = &Q;
	QD = Q.get_QD();
	clone_psvm(p, p_, count_instance);
	clone_psvm(y, y_, count_instance);
	clone_psvm(alpha, alpha_, count_instance);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[count_instance];
		for (int i = 0; i < count_instance; i++) {
			update_alpha_status(i);
		}
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[count_instance];
		for (int i = 0; i < count_instance; i++) {
			active_set[i] = i;
		}
		active_size = count_instance;
	}

	// initialize gradient
	{
		G = new double[count_instance];
		G_bar = new double[count_instance];
		int i;
		for (i = 0; i < count_instance; i++) {
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for (i = 0; i < count_instance; i++)
		if (!is_lower_bound(i)) {
			const Qfloat *Q_i = Q.get_Q(i, count_instance);
			double alpha_i = alpha[i];
			int j;
			for (j = 0; j < count_instance; j++) {
				G[j] += alpha_i*Q_i[j];
			}
			if (is_upper_bound(i)) {
				for (j = 0; j<count_instance; j++) {
					G_bar[j] += get_C(i) * Q_i[j];
				}
			}
		}
	}

	// optimization step

	int iter = 0;
	int max_iter = max_psvm(10000000, count_instance>INT_MAX / 100 ? INT_MAX : 100 * count_instance);
	int counter = min_psvm(count_instance, 1000) + 1;

	while (iter < max_iter) {
		// show progress and do shrinking

		if (--counter == 0) {
			counter = min_psvm(count_instance, 1000);
			if (shrinking) do_shrinking();
			info(".");
		}

		int i, j;
		if (select_working_set(i, j) != 0) {
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = count_instance;
			info("*");
			if (select_working_set(i, j) != 0) {
				break;
			} else {
				counter = 1;	// do shrinking next iteration
			}
		}

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

		const Qfloat *Q_i = Q.get_Q(i, active_size);
		const Qfloat *Q_j = Q.get_Q(j, active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if (y[i] != y[j]) {
			double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i] - G[j]) / quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;

			if (diff > 0) {
				if (alpha[j] < 0) {
					alpha[j] = 0;
					alpha[i] = diff;
				}
			} else {
				if (alpha[i] < 0) {
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if (diff > C_i - C_j) {
				if (alpha[i] > C_i) {
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			} else {
				if (alpha[j] > C_j) {
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		} else {
			double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i] - G[j]) / quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if (sum > C_i) {
				if (alpha[i] > C_i) {
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			} else {
				if (alpha[j] < 0) {
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if (sum > C_j) {
				if (alpha[j] > C_j) {
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			} else {
				if (alpha[i] < 0) {
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		} 

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;

		for (int k = 0; k < active_size; k++) {
			G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
		}

		// update alpha_status and G_bar
		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if (ui != is_upper_bound(i)) {
				Q_i = Q.get_Q(i, count_instance);
				if (ui) {
					for (k = 0; k < count_instance; k++) {
						G_bar[k] -= C_i * Q_i[k];
					}
				}
				else {
					for (k = 0; k < count_instance; k++) {
						G_bar[k] += C_i * Q_i[k];
					}
				}
			}

			if (uj != is_upper_bound(j)) {
				Q_j = Q.get_Q(j, count_instance);
				if (uj) {
					for (k = 0; k < count_instance; k++) {
						G_bar[k] -= C_j * Q_j[k];
					}
				}
				else {
					for (k = 0; k < count_instance; k++) {
						G_bar[k] += C_j * Q_j[k];
					}
				}
			}
		}
	} //end while(iter < max_iter)

	if (iter >= max_iter) {
		if (active_size < count_instance) {
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = count_instance;
			info("*");
		}
		fprintf(stderr, "\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for (i = 0; i < count_instance; i++) {
			v += alpha[i] * (G[i] + p[i]);
		}
		si->obj = v / 2;
	}

	// put back the solution
	{
		for (int i = 0; i < count_instance; i++) {
			alpha_[active_set[i]] = alpha[i];
		}
	}

	// juggle everything back
	/*{
		for(int i=0;i<count_instance;i++)
		while(active_set[i] != i)
		swap_index(i,active_set[i]);
		// or Q.swap_index(i,active_set[i]);
		}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n", iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j) {
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for (int t = 0; t < active_size; t++) {
		if (y[t] == +1) {
			if (!is_upper_bound(t))
			if (-G[t] >= Gmax) {
				Gmax = -G[t];
				Gmax_idx = t;
			}
		} else {
			if (!is_lower_bound(t))
			if (G[t] >= Gmax) {
				Gmax = G[t];
				Gmax_idx = t;
			}
		}
	}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if (i != -1) { // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i, active_size);
	}

	for (int j = 0; j < active_size; j++) {
		if (y[j] == +1) {
			if (!is_lower_bound(j)) {
				double grad_diff = Gmax + G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0) {
					double obj_diff;
					double quad_coef = QD[i] + QD[j] - 2.0*y[i] * Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff) / quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff) / TAU;

					if (obj_diff <= obj_diff_min) {
						Gmin_idx = j;
						obj_diff_min = obj_diff;
					}
				}
			}
		} else {
			if (!is_upper_bound(j)) {
				double grad_diff = Gmax - G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0) {
					double obj_diff;
					double quad_coef = QD[i] + QD[j] + 2.0*y[i] * Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff) / quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff) / TAU;

					if (obj_diff <= obj_diff_min) {
						Gmin_idx = j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if (Gmax + Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2) {
	if (is_upper_bound(i)) {
		if (y[i] == +1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	} else if (is_lower_bound(i)) {
		if (y[i] == +1)
			return(G[i] > Gmax2);
		else
			return(G[i] > Gmax1);
	} else
		return(false);
}

void Solver::do_shrinking() {
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for (i = 0; i < active_size; i++) {
		if (y[i] == +1) {
			if (!is_upper_bound(i)) {
				if (-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if (!is_lower_bound(i)) {
				if (G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		} else {
			if (!is_upper_bound(i)) {
				if (-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if (!is_lower_bound(i)) {
				if (G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if (unshrink == false && Gmax1 + Gmax2 <= eps * 10) {
		unshrink = true;
		reconstruct_gradient();
		active_size = count_instance;
		info("*");
	}

	for (i = 0; i<active_size; i++) {
		if (be_shrunk(i, Gmax1, Gmax2)) {
			active_size--;
			while (active_size > i) {
				if (!be_shrunk(active_size, Gmax1, Gmax2)) {
					swap_index(i, active_size); 
					break;
				}
				active_size--;
			}
		}
	}
}

double Solver::calculate_rho() {
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for (int i = 0; i < active_size; i++) {
		double yG = y[i] * G[i];

		if (is_upper_bound(i)) {
			if (y[i] == -1)
				ub = min_psvm(ub, yG);
			else
				lb = max_psvm(lb, yG);
		} else if (is_lower_bound(i)) {
			if (y[i] == +1)
				ub = min_psvm(ub, yG);
			else
				lb = max_psvm(lb, yG);
		} else {
			++nr_free;
			sum_free += yG;
		}
	}

	if (nr_free > 0)
		r = sum_free / nr_free;
	else
		r = (ub + lb) / 2;

	return r;
}

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j) {
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for (int t = 0; t < active_size; t++)
	if (y[t] == +1) {
		if (!is_upper_bound(t))
		if (-G[t] >= Gmaxp) {
			Gmaxp = -G[t];
			Gmaxp_idx = t;
		}
	} else {
		if (!is_lower_bound(t))
		if (G[t] >= Gmaxn) {
			Gmaxn = G[t];
			Gmaxn_idx = t;
		}
	}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if (ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip, active_size);
	if (in != -1)
		Q_in = Q->get_Q(in, active_size);

	for (int j = 0; j<active_size; j++) {
		if (y[j] == +1) {
			if (!is_lower_bound(j)) {
				double grad_diff = Gmaxp + G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0) {
					double obj_diff;
					double quad_coef = QD[ip] + QD[j] - 2 * Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff) / quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff) / TAU;

					if (obj_diff <= obj_diff_min) {
						Gmin_idx = j;
						obj_diff_min = obj_diff;
					}
				}
			}
		} else {
			if (!is_upper_bound(j)) {
				double grad_diff = Gmaxn - G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0) {
					double obj_diff;
					double quad_coef = QD[in] + QD[j] - 2 * Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff) / quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff) / TAU;

					if (obj_diff <= obj_diff_min) {
						Gmin_idx = j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if (max_psvm(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4) {
	if (is_upper_bound(i)) {
		if (y[i] == +1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax4);
	} else if (is_lower_bound(i)) {
		if (y[i] == +1)
			return(G[i] > Gmax2);
		else
			return(G[i] > Gmax3);
	} else
		return(false);
}

void Solver_NU::do_shrinking() {
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for (i = 0; i<active_size; i++) {
		if (!is_upper_bound(i)) {
			if (y[i] == +1) {
				if (-G[i] > Gmax1) Gmax1 = -G[i];
			} else	if (-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if (!is_lower_bound(i)) {
			if (y[i] == +1) {
				if (G[i] > Gmax2) Gmax2 = G[i];
			} else	if (G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if (unshrink == false && max_psvm(Gmax1 + Gmax2, Gmax3 + Gmax4) <= eps * 10) {
		unshrink = true;
		reconstruct_gradient();
		active_size = count_instance;
	}

	for (i = 0; i<active_size; i++)
	if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)) {
		active_size--;
		while (active_size > i) {
			if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4)) {
				swap_index(i, active_size);
				break;
			}
			active_size--;
		}
	}
}

double Solver_NU::calculate_rho() {
	int nr_free1 = 0, nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for (int i = 0; i < active_size; i++) {
		if (y[i] == +1) {
			if (is_upper_bound(i))
				lb1 = max_psvm(lb1, G[i]);
			else if (is_lower_bound(i))
				ub1 = min_psvm(ub1, G[i]);
			else {
				++nr_free1;
				sum_free1 += G[i];
			}
		} else {
			if (is_upper_bound(i))
				lb2 = max_psvm(lb2, G[i]);
			else if (is_lower_bound(i))
				ub2 = min_psvm(ub2, G[i]);
			else {
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1, r2;
	if (nr_free1 > 0)
		r1 = sum_free1 / nr_free1;
	else
		r1 = (ub1 + lb1) / 2;

	if (nr_free2 > 0)
		r2 = sum_free2 / nr_free2;
	else
		r2 = (ub2 + lb2) / 2;

	si->r = (r1 + r2) / 2;
	return (r1 - r2) / 2;
}

Solver_NU::Solver_NU() {
}

void Solver_NU::Solve(int count_instance, const QMatrix& Q, const double *p, const schar *y, double *alpha, double Cp, double Cn, double eps, SolutionInfo* si, int shrinking) {
	this->si = si;
	Solver::Solve(count_instance, Q, p, y, alpha, Cp, Cn, eps, si, shrinking);
}

SVC_Q::SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_) 
	: Kernel(prob.node, prob.count_instance, prob.count_attribute, prob.elements, prob.indexes, param) {		
	
		clone_psvm(y, y_, prob.count_instance);
		cache = new Cache(prob.count_instance, (long int)(param.cache_size*(1 << 20)));
		QD = new double [prob.count_instance];
		
		switch(kernel_mode) {
			//*
			case _KERNEL_RBF: {
				//ofstream myfile;
				//myfile.open("parallel.txt");
				
				gpusocket->run(prob.count_attribute, prob.count_instance, &aux_buffer[0], NULL, DIAGONAL);
				for(int n = 0; n < prob.count_instance; n++) {
					QD[n] = exp(-gamma*(x_square[n] + x_square[n] - 2 * aux_buffer[prob.indexes[n]]));

					//myfile << QD[n] << endl;
				}

				//myfile.close();

				break;
			}
			//*/
			default: { //default case for not improved yet methods
				for (int i = 0; i < prob.count_instance; i++) {
					QD[i] = (this->*kernel_function)(i, i);
				}
				break;
			}
		}
	}

void SVC_Q::swap_index(int i, int j) const {
		cache->swap_index(i, j);
		Kernel::swap_index(i, j);
		swap_psvm(y[i], y[j]);
		swap_psvm(QD[i], QD[j]);
	}

Qfloat *SVC_Q::get_Q(int i, int len) const {
		int start, j;
		Qfloat *data = NULL;
		
		if((start = cache->get_data(i, &data, len)) < len) {
			//original 2
			switch(kernel_mode) {
				case _KERNEL_RBF: {
					gpusocket->run(count_attribute, count_instance, &aux_buffer[0], indexes[i], ONE_VERSUS_ALL);
					for(j = start; j < len; j++) {
						data[j] = (Qfloat)(y[i] * y[j] * exp(-gamma*(x_square[i] + x_square[j] - 2 * (double)aux_buffer[indexes[j]])));

						//TODO REMOVE LATER
						/*
						Qfloat original = (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));
						if(original != data[j]) {
							cout << "error!" << endl;
						}
						*/
						//TODO REMOVE LATER

					}
					break;
				}
				default: { //for not improved yet kernels
					for (j = start; j < len; j++) {
						data[j] = (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));
					}	
					break;
				}
			}
		}
		return data;
	}

double *SVC_Q::get_QD() const {
	return QD;
}

SVC_Q::~SVC_Q() {
	delete[] y;
	delete cache;
	delete[] QD;

	//wizard->releaseClassBuffer();
}

ONE_CLASS_Q::ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	: Kernel(prob.node, prob.count_instance, prob.count_attribute, prob.elements, prob.indexes, param) {
		cache = new Cache(prob.count_instance, (long int)(param.cache_size*(1 << 20)));
		QD = new double[prob.count_instance];
		for (int i = 0; i < prob.count_instance; i++)
			QD[i] = (this->*kernel_function)(i, i);
	}

Qfloat *ONE_CLASS_Q::get_Q(int i, int len) const {
	Qfloat *data;
	int start, j;
	if ((start = cache->get_data(i, &data, len)) < len) {
		for (j = start; j < len; j++) {
			data[j] = (Qfloat)(this->*kernel_function)(i, j);
		}
	}
	return data;
}

double *ONE_CLASS_Q::get_QD() const {
	return QD;
}

void ONE_CLASS_Q::swap_index(int i, int j) const {
	cache->swap_index(i, j);
	Kernel::swap_index(i, j);
	swap_psvm(QD[i], QD[j]);
}

ONE_CLASS_Q::~ONE_CLASS_Q() {
	delete cache;
	delete[] QD;
}

SVR_Q::SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.node, prob.count_instance, prob.count_attribute, prob.elements, prob.indexes, param) {
	count_instance = prob.count_instance;
	cache = new Cache(count_instance, (long int)(param.cache_size*(1 << 20)));
	QD = new double[2 * count_instance];
	sign = new schar[2 * count_instance];
	index = new int[2 * count_instance];
	for (int k = 0; k < count_instance; k++) {
		sign[k] = 1;
		sign[k + count_instance] = -1;
		index[k] = k;
		index[k + count_instance] = k;
		QD[k] = (this->*kernel_function)(k, k);
		QD[k + count_instance] = QD[k];
	}
	buffer[0] = new Qfloat[2 * count_instance];
	buffer[1] = new Qfloat[2 * count_instance];
	next_buffer = 0;
}

void SVR_Q::swap_index(int i, int j) const {
	swap_psvm(sign[i], sign[j]);
	swap_psvm(index[i], index[j]);
	swap_psvm(QD[i], QD[j]);
}

Qfloat *SVR_Q::get_Q(int i, int len) const {
	Qfloat *data;
	int j, real_i = index[i];
	if (cache->get_data(real_i, &data, count_instance) < count_instance) {
		for (j = 0; j < count_instance; j++) {
			data[j] = (Qfloat)(this->*kernel_function)(real_i, j);
		}
	}

	// reorder and copy
	Qfloat *buf = buffer[next_buffer];
	next_buffer = 1 - next_buffer;
	schar si = sign[i];
	for (j = 0; j < len; j++)
		buf[j] = (Qfloat)si * (Qfloat)sign[j] * data[index[j]];
	return buf;
}

double *SVR_Q::get_QD() const {
	return QD;
}

SVR_Q::~SVR_Q() {
	delete cache;
	delete[] sign;
	delete[] index;
	delete[] buffer[0];
	delete[] buffer[1];
	delete[] QD;
}

//
// construct and solve various formulations
//
static void solve_c_svc(const svm_problem *prob, const svm_parameter* param, double *alpha, Solver::SolutionInfo* si, double Cp, double Cn) {
	
	int count_instance = prob->count_instance;
	double *minus_ones = new double [count_instance];
	schar *y = new schar [count_instance];

	int i;

	for (i = 0; i <count_instance; i++) {
		alpha[i] = 0;
		minus_ones[i] = -1;
		if (prob->class_value[i] > 0) {
			y[i] = +1; 
		} else {
			y[i] = -1;
		}
	}

	Solver s;
	s.Solve(count_instance, SVC_Q(*prob, *param, y), minus_ones, y, alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for (i = 0; i < count_instance; i++)
		sum_alpha += alpha[i];

	if (Cp == Cn)
		info("nu = %f\n", sum_alpha / (Cp*prob->count_instance));

	for (i = 0; i < count_instance; i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si) {
	int i;
	int count_instance = prob->count_instance;
	double nu = param->nu;

	schar *y = new schar[count_instance];

	for (i = 0; i<count_instance; i++)
	if (prob->class_value[i]>0)
		y[i] = +1;
	else
		y[i] = -1;

	double sum_pos = nu*count_instance / 2;
	double sum_neg = nu*count_instance / 2;

	for (i = 0; i < count_instance; i++)
	if (y[i] == +1) {
		alpha[i] = min_psvm(1.0, sum_pos);
		sum_pos -= alpha[i];
	} else {
		alpha[i] = min_psvm(1.0, sum_neg);
		sum_neg -= alpha[i];
	}

	double *zeros = new double[count_instance];

	for (i = 0; i < count_instance; i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(count_instance, SVC_Q(*prob, *param, y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);
	double r = si->r;

	info("C = %f\n", 1 / r);

	for (i = 0; i < count_instance; i++)
		alpha[i] *= y[i] / r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1 / r;
	si->upper_bound_n = 1 / r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si) {
	int count_instance = prob->count_instance;
	double *zeros = new double[count_instance];
	schar *ones = new schar[count_instance];
	int i;

	int n = (int)(param->nu*prob->count_instance);	// # of alpha's at upper bound

	for (i = 0; i < n; i++)
		alpha[i] = 1;
	if (n < prob->count_instance)
		alpha[n] = param->nu * prob->count_instance - n;
	for (i = n + 1; i < count_instance; i++)
		alpha[i] = 0;

	for (i = 0; i < count_instance; i++) {
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(count_instance, ONE_CLASS_Q(*prob, *param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si) {
	int count_instance = prob->count_instance;
	double *alpha2 = new double[2 * count_instance];
	double *linear_term = new double[2 * count_instance];
	schar *y = new schar[2 * count_instance];
	int i;

	for (i = 0; i < count_instance; i++) {
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->class_value[i];
		y[i] = 1;

		alpha2[i + count_instance] = 0;
		linear_term[i + count_instance] = param->p + prob->class_value[i];
		y[i + count_instance] = -1;
	}

	Solver s;
	s.Solve(2 * count_instance, SVR_Q(*prob, *param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for (i = 0; i < count_instance; i++) {
		alpha[i] = alpha2[i] - alpha2[i + count_instance];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n", sum_alpha / (param->C*count_instance));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si) {
	int count_instance = prob->count_instance;
	double C = param->C;
	double *alpha2 = new double[2 * count_instance];
	double *linear_term = new double[2 * count_instance];
	schar *y = new schar[2 * count_instance];
	int i;

	double sum = C * param->nu * count_instance / 2;
	for (i = 0; i < count_instance; i++) {
		alpha2[i] = alpha2[i + count_instance] = min_psvm(sum, C);
		sum -= alpha2[i];

		linear_term[i] = -prob->class_value[i];
		y[i] = 1;

		linear_term[i + count_instance] = prob->class_value[i];
		y[i + count_instance] = -1;
	}

	Solver_NU s;
	s.Solve(2 * count_instance, SVR_Q(*prob, *param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n", -si->r);

	for (i = 0; i < count_instance; i++)
		alpha[i] = alpha2[i] - alpha2[i + count_instance];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static decision_function svm_train_one(const svm_problem *prob, const svm_parameter *param, double Cp, double Cn) {
	
	double *alpha = Malloc(double, prob->count_instance);
	Solver::SolutionInfo si;
	
	switch (param->svm_type) {
		case C_SVC:
		solve_c_svc(prob, param, alpha, &si, Cp, Cn);
		break;
		case NU_SVC:
		solve_nu_svc(prob, param, alpha, &si);
		break;
		case ONE_CLASS:
		solve_one_class(prob, param, alpha, &si);
		break;
		case EPSILON_SVR:
		solve_epsilon_svr(prob, param, alpha, &si);
		break;
		case NU_SVR:
		solve_nu_svr(prob, param, alpha, &si);
		break;
	}

	info("obj = %f, rho = %f\n", si.obj, si.rho);

	// output SVs

	int 
		nSV = 0,
		nBSV = 0;
	for (int i = 0; i < prob->count_instance; i++) {
		if (fabs(alpha[i]) > 0) {
			++nSV;
			if (prob->class_value[i] > 0) {
				if (fabs(alpha[i]) >= si.upper_bound_p) {
					++nBSV;
				}
			} else {
				if (fabs(alpha[i]) >= si.upper_bound_n) {
					++nBSV;
				}
			}
		}
	}

	info("nSV = %d, nBSV = %d\n", nSV, nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int count_instance, const double *dec_values, const double *labels,
	double& A, double& B) {
	double prior1 = 0, prior0 = 0;
	int i;

	for (i = 0; i<count_instance; i++)
	if (labels[i] > 0) prior1 += 1;
	else prior0 += 1;

	int max_iter = 100;	// Maximal number of iterations
	double min_step = 1e-10;	// Minimal step taken in line search
	double sigma = 1e-12;	// For numerically strict PD of Hessian
	double eps = 1e-5;
	double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
	double loTarget = 1 / (prior0 + 2.0);
	double *t = Malloc(double, count_instance);
	double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
	double newA, newB, newf, d1, d2;
	int iter;

	// Initial Point and Initial Fun Value
	A = 0.0; B = log((prior0 + 1.0) / (prior1 + 1.0));
	double fval = 0.0;

	for (i = 0; i<count_instance; i++) {
		if (labels[i]>0) t[i] = hiTarget;
		else t[i] = loTarget;
		fApB = dec_values[i] * A + B;
		if (fApB >= 0)
			fval += t[i] * fApB + log(1 + exp(-fApB));
		else
			fval += (t[i] - 1)*fApB + log(1 + exp(fApB));
	}
	for (iter = 0; iter < max_iter; iter++) {
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11 = sigma; // numerically ensures strict PD
		h22 = sigma;
		h21 = 0.0; g1 = 0.0; g2 = 0.0;
		for (i = 0; i < count_instance; i++) {
			fApB = dec_values[i] * A + B;
			if (fApB >= 0) {
				p = exp(-fApB) / (1.0 + exp(-fApB));
				q = 1.0 / (1.0 + exp(-fApB));
			} else {
				p = 1.0 / (1.0 + exp(fApB));
				q = exp(fApB) / (1.0 + exp(fApB));
			}
			d2 = p*q;
			h11 += dec_values[i] * dec_values[i] * d2;
			h22 += d2;
			h21 += dec_values[i] * d2;
			d1 = t[i] - p;
			g1 += dec_values[i] * d1;
			g2 += d1;
		}

		// Stopping Criteria
		if (fabs(g1) < eps && fabs(g2) < eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det = h11*h22 - h21*h21;
		dA = -(h22*g1 - h21 * g2) / det;
		dB = -(-h21*g1 + h11 * g2) / det;
		gd = g1*dA + g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step) {
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i = 0; i < count_instance; i++) {
				fApB = dec_values[i] * newA + newB;
				if (fApB >= 0)
					newf += t[i] * fApB + log(1 + exp(-fApB));
				else
					newf += (t[i] - 1)*fApB + log(1 + exp(fApB));
			}
			// Check sufficient decrease
			if (newf < fval + 0.0001*stepsize*gd) {
				A = newA; B = newB; fval = newf;
				break;
			} else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step) {
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter >= max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B) {
	double fApB = decision_value*A + B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB) / (1.0 + exp(-fApB));
	else
		return 1.0 / (1 + exp(fApB));
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p) {
	int t, j;
	int iter = 0, max_iter = max_psvm(100, k);
	double **Q = Malloc(double *, k);
	double *Qp = Malloc(double, k);
	double pQp, eps = 0.005 / k;

	for (t = 0; t < k; t++) {
		p[t] = 1.0 / k;  // Valid if k = 1
		Q[t] = Malloc(double, k);
		Q[t][t] = 0;
		for (j = 0; j < t; j++) {
			Q[t][t] += r[j][t] * r[j][t];
			Q[t][j] = Q[j][t];
		}
		for (j = t + 1; j < k; j++) {
			Q[t][t] += r[j][t] * r[j][t];
			Q[t][j] = -r[j][t] * r[t][j];
		}
	}
	for (iter = 0; iter < max_iter; iter++) {
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0;
		for (t = 0; t < k; t++) {
			Qp[t] = 0;
			for (j = 0; j < k; j++)
				Qp[t] += Q[t][j] * p[j];
			pQp += p[t] * Qp[t];
		}
		double max_error = 0;
		for (t = 0; t<k; t++) {
			double error = fabs(Qp[t] - pQp);
			if (error>max_error)
				max_error = error;
		}
		if (max_error < eps) break;

		for (t = 0; t < k; t++) {
			double diff = (-Qp[t] + pQp) / Q[t][t];
			p[t] += diff;
			pQp = (pQp + diff*(diff*Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
			for (j = 0; j < k; j++) {
				Qp[j] = (Qp[j] + diff*Q[t][j]) / (1 + diff);
				p[j] /= (1 + diff);
			}
		}
	}
	if (iter >= max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for (t = 0; t < k; t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB) {
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int, prob->count_instance);
	double *dec_values = Malloc(double, prob->count_instance);

	// random shuffle
	for (i = 0; i < prob->count_instance; i++) perm[i] = i;
	for (i = 0; i < prob->count_instance; i++) {
		int j = i + rand() % (prob->count_instance - i);
		swap_psvm(perm[i], perm[j]);
	}
	for (i = 0; i < nr_fold; i++) {
		int begin = i*prob->count_instance / nr_fold;
		int end = (i + 1)*prob->count_instance / nr_fold;
		int j, k;
		struct svm_problem subprob;

		subprob.count_instance = prob->count_instance - (end - begin);
		subprob.node = Malloc(svm_node*, subprob.count_instance);
		subprob.class_value = Malloc(double, subprob.count_instance);

		k = 0;
		for (j = 0; j < begin; j++) {
			subprob.node[k] = prob->node[perm[j]];
			subprob.class_value[k] = prob->class_value[perm[j]];
			++k;
		}
		for (j = end; j < prob->count_instance; j++) {
			subprob.node[k] = prob->node[perm[j]];
			subprob.class_value[k] = prob->class_value[perm[j]];
			++k;
		}
		int p_count = 0, n_count = 0;
		for (j = 0; j<k; j++)
		if (subprob.class_value[j]>0)
			p_count++;
		else
			n_count++;

		if (p_count == 0 && n_count == 0)
		for (j = begin; j<end; j++)
			dec_values[perm[j]] = 0;
		else if (p_count > 0 && n_count == 0)
		for (j = begin; j<end; j++)
			dec_values[perm[j]] = 1;
		else if (p_count == 0 && n_count > 0)
		for (j = begin; j < end; j++)
			dec_values[perm[j]] = -1;
		else {
			svm_parameter subparam = *param;
			subparam.probability = 0;
			subparam.C = 1.0;
			subparam.nr_weight = 2;
			subparam.weight_label = Malloc(int, 2);
			subparam.weight = Malloc(double, 2);
			subparam.weight_label[0] = +1;
			subparam.weight_label[1] = -1;
			subparam.weight[0] = Cp;
			subparam.weight[1] = Cn;
			struct svm_model *submodel = svm_train(&subprob, &subparam);
			for (j = begin; j < end; j++) {
				svm_predict_values(submodel, prob->node[perm[j]], &(dec_values[perm[j]]));
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.node);
		free(subprob.class_value);
	}
	sigmoid_train(prob->count_instance, dec_values, prob->class_value, probA, probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param) {
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double, prob->count_instance);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob, &newparam, nr_fold, ymv);
	for (i = 0; i < prob->count_instance; i++) {
		ymv[i] = prob->class_value[i] - ymv[i];
		mae += fabs(ymv[i]);
	}
	mae /= prob->count_instance;
	double std = sqrt(2 * mae*mae);
	int count = 0;
	mae = 0;
	for (i = 0; i<prob->count_instance; i++)
	if (fabs(ymv[i]) > 5 * std)
		count = count + 1;
	else
		mae += fabs(ymv[i]);
	mae /= (prob->count_instance - count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n", mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length count_instance, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm) {
	int count_instance = prob->count_instance;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int, max_nr_class);
	int *count = Malloc(int, max_nr_class);
	int *data_label = Malloc(int, count_instance);
	int i;

	for (i = 0; i < count_instance; i++) {
		int this_label = (int)prob->class_value[i];
		int j;
		for (j = 0; j < nr_class; j++) {
			if (this_label == label[j]) {
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if (j == nr_class) {
			if (nr_class == max_nr_class) {
				max_nr_class *= 2;
				label = (int *)realloc(label, max_nr_class*sizeof(int));
				count = (int *)realloc(count, max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1) {
		swap_psvm(label[0], label[1]);
		swap_psvm(count[0], count[1]);
		for (i = 0; i < count_instance; i++) {
			if (data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int, nr_class);
	start[0] = 0;
	for (i = 1; i < nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];
	for (i = 0; i < count_instance; i++) {
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for (i = 1; i < nr_class; i++)
		start[i] = start[i - 1] + count[i - 1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param) {
	svm_model *model = Malloc(svm_model, 1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if (param->svm_type == ONE_CLASS ||
		param->svm_type == EPSILON_SVR ||
		param->svm_type == NU_SVR) {
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *, 1);

		if (param->probability &&
			(param->svm_type == EPSILON_SVR ||
			param->svm_type == NU_SVR)) {
			model->probA = Malloc(double, 1);
			model->probA[0] = svm_svr_probability(prob, param);
		}

		decision_function f = svm_train_one(prob, param, 0, 0);
		model->rho = Malloc(double, 1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for (i = 0; i < prob->count_instance; i++) {
			if (fabs(f.alpha[i]) > 0) {
				++nSV;
			}
		}
		model->count_instance = nSV;
		model->SV = Malloc(svm_node *, nSV);
		model->sv_coef[0] = Malloc(double, nSV);
		model->sv_indices = Malloc(int, nSV);
		int j = 0;
		for (i = 0; i < prob->count_instance; i++) {
			if (fabs(f.alpha[i]) > 0) {
				model->SV[j] = prob->node[i]; //TODO verify!
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i + 1;
				++j;
			}
		}

		free(f.alpha);
	} else {
		// classification
		int count_instance = prob->count_instance;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int, count_instance); //perm has the index of positive and negative attributes

		// group training data of the same class
		svm_group_classes(prob, &nr_class, &label, &start, &count, perm); 

		if (nr_class == 1) {
			info("WARNING: training data in only one class. See README for details.\n");
		}

		svm_node **node = Malloc(svm_node *, count_instance); //TODO commented
		int *indexes = new int [count_instance];

		int i;
		for (i = 0; i < count_instance; i++) {
			node[i] = prob->node[perm[i]]; //TODO commented
			indexes[i] = prob->indexes[perm[i]]; 
		}

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for (i = 0; i < nr_class; i++) {
			weighted_C[i] = param->C;
		}
		for (i = 0; i < param->nr_weight; i++) {
			int j;
			for (j = 0; j < nr_class; j++)
			if (param->weight_label[i] == label[j]) {
				break;
			}
			if (j == nr_class) {
				fprintf(stderr, "WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			} else {
				weighted_C[j] *= param->weight[i];
			}
		}

		// train k*(k-1)/2 models

		bool *nonzero = Malloc(bool, count_instance);
		for (i = 0; i < count_instance; i++) {
			nonzero[i] = false;
		}
		decision_function *f = Malloc(decision_function, nr_class*(nr_class - 1) / 2);

		double *probA = NULL, *probB = NULL;
		if (param->probability) {
			probA = Malloc(double, nr_class*(nr_class - 1) / 2);
			probB = Malloc(double, nr_class*(nr_class - 1) / 2);
		}

		int p = 0;
		for (i = 0; i < nr_class; i++) {
			for (int j = i + 1; j < nr_class; j++) {
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.count_instance = ci + cj;
				sub_prob.elements = prob->elements;
				sub_prob.node = Malloc(svm_node *, sub_prob.count_instance); //TODO commented
				sub_prob.indexes = Malloc(int, sub_prob.count_instance);
				sub_prob.class_value = Malloc(double, sub_prob.count_instance);
				sub_prob.count_attribute = prob->count_attribute;
				
				int k;
				for (k = 0; k < ci; k++) {
					sub_prob.node[k] = node[si + k]; //TODO commented
					sub_prob.indexes[k] = indexes[si + k]; //TODO inserted
					sub_prob.class_value[k] = +1;
				}
				for (k = 0; k < cj; k++) {
					sub_prob.node[ci + k] = node[sj + k]; //TODO commented
					sub_prob.indexes[ci + k] = indexes[sj + k]; //TODO inserted
					sub_prob.class_value[ci + k] = -1;
				}

				if (param->probability) {
					svm_binary_svc_probability(&sub_prob, param, weighted_C[i], weighted_C[j], probA[p], probB[p]);
				}

				f[p] = svm_train_one(&sub_prob, param, weighted_C[i], weighted_C[j]);
				for (k = 0; k<ci; k++) {
					if (!nonzero[si + k] && fabs(f[p].alpha[k]) > 0) {
						nonzero[si + k] = true;
					}
				}
				for (k = 0; k<cj; k++) {
					if (!nonzero[sj + k] && fabs(f[p].alpha[ci + k]) > 0) {
						nonzero[sj + k] = true;
					}
				}
				free(sub_prob.node); //TODO commented
				free(sub_prob.class_value);
				free(sub_prob.indexes);
				++p;
			}
		}

		// build output

		model->nr_class = nr_class;

		model->label = Malloc(int, nr_class);
		for (i = 0; i < nr_class; i++) {
			model->label[i] = label[i];
		}

		model->rho = Malloc(double, nr_class*(nr_class - 1) / 2);
		for (i = 0; i < nr_class*(nr_class - 1) / 2; i++) {
			model->rho[i] = f[i].rho;
		}

		if (param->probability) {
			model->probA = Malloc(double, nr_class*(nr_class - 1) / 2);
			model->probB = Malloc(double, nr_class*(nr_class - 1) / 2);
			for (i = 0; i < nr_class*(nr_class - 1) / 2; i++) {
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		} else {
			model->probA = NULL;
			model->probB = NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int, nr_class);
		model->nSV = Malloc(int, nr_class);
		for (i = 0; i < nr_class; i++) {
			int nSV = 0;
			for (int j = 0; j < count[i]; j++)
			if (nonzero[start[i] + j]) {
				++nSV;
				++total_sv;
			}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		info("Total nSV = %d\n", total_sv);

		model->count_instance = total_sv;
		model->SV = Malloc(svm_node *, total_sv); //TODO commented
		model->sv_indices = Malloc(int, total_sv);
		p = 0;
		for (i = 0; i < count_instance; i++) {
			if (nonzero[i]) {
				model->SV[p] = node[i]; //TODO commented
				model->sv_indices[p++] = perm[i] + 1;
			}
		}

		int *nz_start = Malloc(int, nr_class);
		nz_start[0] = 0;
		for (i = 1; i < nr_class; i++) {
			nz_start[i] = nz_start[i - 1] + nz_count[i - 1];
		}

		model->sv_coef = Malloc(double *, nr_class - 1);
		for (i = 0; i < nr_class - 1; i++) {
			model->sv_coef[i] = Malloc(double, total_sv);
		}

		p = 0;
		for (i = 0; i < nr_class; i++) {
			for (int j = i + 1; j < nr_class; j++) {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];

				int q = nz_start[i];
				int k;
				for (k = 0; k < ci; k++)
				if (nonzero[si + k])
					model->sv_coef[j - 1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for (k = 0; k < cj; k++)
				if (nonzero[sj + k])
					model->sv_coef[i][q++] = f[p].alpha[ci + k];
				++p;
			}
		}

		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(node);
		free(weighted_C);
		free(nonzero);
		for (i = 0; i < nr_class*(nr_class - 1) / 2; i++) {
			free(f[i].alpha);
		}
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target) {
	int i;
	int *fold_start;
	int count_instance = prob->count_instance;
	int *perm = Malloc(int, count_instance);
	int nr_class;
	if (nr_fold > count_instance) {
		nr_fold = count_instance;
		fprintf(stderr, "WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int, nr_fold + 1);
	// stratified cv may not give leave-one-out rate
	// Each class to count_instance folds -> some folds may have zero elements
	if ((param->svm_type == C_SVC ||
		param->svm_type == NU_SVC) && nr_fold < count_instance) {
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob, &nr_class, &label, &start, &count, perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int, nr_fold);
		int c;
		int *index = Malloc(int, count_instance);
		for (i = 0; i < count_instance; i++)
			index[i] = perm[i];
		for (c = 0; c < nr_class; c++)
		for (i = 0; i < count[c]; i++) {
			int j = i + rand() % (count[c] - i);
			swap_psvm(index[start[c] + j], index[start[c] + i]);
		}
		for (i = 0; i < nr_fold; i++) {
			fold_count[i] = 0;
			for (c = 0; c < nr_class; c++)
				fold_count[i] += (i + 1)*count[c] / nr_fold - i*count[c] / nr_fold;
		}
		fold_start[0] = 0;
		for (i = 1; i <= nr_fold; i++)
			fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		for (c = 0; c < nr_class; c++)
		for (i = 0; i < nr_fold; i++) {
			int begin = start[c] + i*count[c] / nr_fold;
			int end = start[c] + (i + 1)*count[c] / nr_fold;
			for (int j = begin; j < end; j++) {
				perm[fold_start[i]] = index[j];
				fold_start[i]++;
			}
		}
		fold_start[0] = 0;
		for (i = 1; i <= nr_fold; i++)
			fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	} else {
		for (i = 0; i < count_instance; i++) perm[i] = i;
		for (i = 0; i < count_instance; i++) {
			int j = i + rand() % (count_instance - i);
			swap_psvm(perm[i], perm[j]);
		}
		for (i = 0; i <= nr_fold; i++)
			fold_start[i] = i*count_instance / nr_fold;
	}

	for (i = 0; i < nr_fold; i++) {
		int begin = fold_start[i];
		int end = fold_start[i + 1];
		int j, k;
		struct svm_problem subprob;

		subprob.count_instance = count_instance - (end - begin);
		subprob.node = Malloc(svm_node*, subprob.count_instance);
		subprob.class_value = Malloc(double, subprob.count_instance);
		subprob.indexes = Malloc(int, subprob.count_instance); //TODO inserted

		k = 0;
		for (j = 0; j < begin; j++) {
			subprob.node[k] = prob->node[perm[j]];
			subprob.class_value[k] = prob->class_value[perm[j]];
			subprob.indexes[k] = prob->indexes[perm[j]]; //TODO inserted
			++k;
		}
		for (j = end; j < count_instance; j++) {
			subprob.node[k] = prob->node[perm[j]];
			subprob.class_value[k] = prob->class_value[perm[j]];
			subprob.indexes[k] = prob->indexes[perm[j]]; //TODO inserted
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob, param);
		if (param->probability &&
			(param->svm_type == C_SVC || param->svm_type == NU_SVC)) {
			double *prob_estimates = Malloc(double, svm_get_nr_class(submodel));
			for (j = begin; j < end; j++)
				target[perm[j]] = svm_predict_probability(submodel, prob->node[perm[j]], prob_estimates);
			free(prob_estimates);
		} else
		for (j = begin; j < end; j++)
			target[perm[j]] = svm_predict(submodel, prob->node[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.node);
		free(subprob.class_value);
		free(subprob.indexes); //TODO inserted
	}
	free(fold_start);
	free(perm);
}


int svm_get_svm_type(const svm_model *model) {
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model) {
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label) {
	if (model->label != NULL)
	for (int i = 0; i < model->nr_class; i++)
		label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices) {
	if (model->sv_indices != NULL)
	for (int i = 0; i < model->count_instance; i++)
		indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model) {
	return model->count_instance;
}

double svm_get_svr_probability(const svm_model *model) {
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		model->probA != NULL)
		return model->probA[0];
	else {
		fprintf(stderr, "Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double svm_predict_values(const svm_model *model, const svm_node *node, double* dec_values) {
	int i;
	if (model->param.svm_type == ONE_CLASS ||
		model->param.svm_type == EPSILON_SVR ||
		model->param.svm_type == NU_SVR) {
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for (i = 0; i<model->count_instance; i++)
			sum += sv_coef[i] * Kernel::k_function(node, model->SV[i], model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if (model->param.svm_type == ONE_CLASS)
			return (sum>0) ? 1 : -1;
		else
			return sum;
	} else {
		int nr_class = model->nr_class;
		int count_instance = model->count_instance;

		double *kvalue = Malloc(double, count_instance);
		for (i = 0; i < count_instance; i++)
			kvalue[i] = Kernel::k_function(node, model->SV[i], model->param);

		int *start = Malloc(int, nr_class);
		start[0] = 0;
		for (i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + model->nSV[i - 1];

		int *vote = Malloc(int, nr_class);
		for (i = 0; i < nr_class; i++)
			vote[i] = 0;

		int p = 0;
		for (i = 0; i < nr_class; i++)
		for (int j = i + 1; j < nr_class; j++) {
			double sum = 0;
			int si = start[i];
			int sj = start[j];
			int ci = model->nSV[i];
			int cj = model->nSV[j];

			int k;
			double *coef1 = model->sv_coef[j - 1];
			double *coef2 = model->sv_coef[i];
			for (k = 0; k < ci; k++)
				sum += coef1[si + k] * kvalue[si + k];
			for (k = 0; k<cj; k++)
				sum += coef2[sj + k] * kvalue[sj + k];
			sum -= model->rho[p];
			dec_values[p] = sum;

			if (dec_values[p] > 0)
				++vote[i];
			else
				++vote[j];
			p++;
		}

		int vote_max_idx = 0;
		for (i = 1; i<nr_class; i++)
		if (vote[i] > vote[vote_max_idx])
			vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *node) {
	int nr_class = model->nr_class;
	double *dec_values;
	if (model->param.svm_type == ONE_CLASS ||
		model->param.svm_type == EPSILON_SVR ||
		model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else
		dec_values = Malloc(double, nr_class*(nr_class - 1) / 2);
	double pred_result = svm_predict_values(model, node, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *node, double *prob_estimates) {
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA != NULL && model->probB != NULL) {
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class - 1) / 2);
		svm_predict_values(model, node, dec_values);

		double min_prob = 1e-7;
		double **pairwise_prob = Malloc(double *, nr_class);
		for (i = 0; i < nr_class; i++)
			pairwise_prob[i] = Malloc(double, nr_class);
		int k = 0;
		for (i = 0; i < nr_class; i++)
		for (int j = i + 1; j < nr_class; j++) {
			pairwise_prob[i][j] = min_psvm(max_psvm(sigmoid_predict(dec_values[k], model->probA[k], model->probB[k]), min_prob), 1 - min_prob);
			pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
			k++;
		}
		multiclass_probability(nr_class, pairwise_prob, prob_estimates);

		int prob_max_idx = 0;
		for (i = 1; i<nr_class; i++)
		if (prob_estimates[i] > prob_estimates[prob_max_idx])
			prob_max_idx = i;
		for (i = 0; i < nr_class; i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	} else
		return svm_predict(model, node);
}

static const char *svm_type_table[] = {
	"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", NULL
};

static const char *kernel_type_table[] = {
	"linear", "polynomial", "rbf", "sigmoid", "precomputed", NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model) {
	FILE *fp = fopen(model_file_name, "w");
	if (fp == NULL) return -1;

	char *old_locale = _strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	const svm_parameter& param = model->param;

	fprintf(fp, "svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp, "kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if (param.kernel_type == POLY)
		fprintf(fp, "degree %d\n", param.degree);

	if (param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp, "gamma %g\n", param.gamma);

	if (param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp, "coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int count_instance = model->count_instance;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n", count_instance);

	{
		fprintf(fp, "rho");
		for (int i = 0; i < nr_class*(nr_class - 1) / 2; i++)
			fprintf(fp, " %g", model->rho[i]);
		fprintf(fp, "\n");
	}

	if (model->label) {
		fprintf(fp, "label");
		for (int i = 0; i < nr_class; i++)
			fprintf(fp, " %d", model->label[i]);
		fprintf(fp, "\n");
	}

	if (model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for (int i = 0; i < nr_class*(nr_class - 1) / 2; i++)
			fprintf(fp, " %g", model->probA[i]);
		fprintf(fp, "\n");
	}
	if (model->probB) {
		fprintf(fp, "probB");
		for (int i = 0; i < nr_class*(nr_class - 1) / 2; i++)
			fprintf(fp, " %g", model->probB[i]);
		fprintf(fp, "\n");
	}

	if (model->nSV) {
		fprintf(fp, "nr_sv");
		for (int i = 0; i < nr_class; i++)
			fprintf(fp, " %d", model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for (int i = 0; i < count_instance; i++) {
		for (int j = 0; j < nr_class - 1; j++)
			fprintf(fp, "%.16g ", sv_coef[j][i]);

		const svm_node *p = SV[i];

		if (param.kernel_type == PRECOMPUTED)
			fprintf(fp, "0:%d ", (int)(p->value));
		else
		while (p->index != -1) {
			fprintf(fp, "%d:%.8g ", p->index, p->value);
			p++;
		}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input) {
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL) {
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

svm_model *svm_load_model(const char *model_file_name) {
	FILE *fp = fopen(model_file_name, "rb");
	if (fp == NULL) return NULL;

	char *old_locale = _strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model, 1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	while (1) {
		fscanf(fp, "%80s", cmd);

		if (strcmp(cmd, "svm_type") == 0) {
			fscanf(fp, "%80s", cmd);
			int i;
			for (i = 0; svm_type_table[i]; i++) {
				if (strcmp(svm_type_table[i], cmd) == 0) {
					param.svm_type = i;
					break;
				}
			}
			if (svm_type_table[i] == NULL) {
				fprintf(stderr, "unknown svm type.\n");

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		} else if (strcmp(cmd, "kernel_type") == 0) {
			fscanf(fp, "%80s", cmd);
			int i;
			for (i = 0; kernel_type_table[i]; i++) {
				if (strcmp(kernel_type_table[i], cmd) == 0) {
					param.kernel_type = i;
					break;
				}
			}
			if (kernel_type_table[i] == NULL) {
				fprintf(stderr, "unknown kernel function.\n");

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		} else if (strcmp(cmd, "degree") == 0)
			fscanf(fp, "%d", &param.degree);
		else if (strcmp(cmd, "gamma") == 0)
			fscanf(fp, "%lf", &param.gamma);
		else if (strcmp(cmd, "coef0") == 0)
			fscanf(fp, "%lf", &param.coef0);
		else if (strcmp(cmd, "nr_class") == 0)
			fscanf(fp, "%d", &model->nr_class);
		else if (strcmp(cmd, "total_sv") == 0)
			fscanf(fp, "%d", &model->count_instance);
		else if (strcmp(cmd, "rho") == 0) {
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->rho = Malloc(double, n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%lf", &model->rho[i]);
		} else if (strcmp(cmd, "label") == 0) {
			int n = model->nr_class;
			model->label = Malloc(int, n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%d", &model->label[i]);
		} else if (strcmp(cmd, "probA") == 0) {
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->probA = Malloc(double, n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%lf", &model->probA[i]);
		} else if (strcmp(cmd, "probB") == 0) {
			int n = model->nr_class * (model->nr_class - 1) / 2;
			model->probB = Malloc(double, n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%lf", &model->probB[i]);
		} else if (strcmp(cmd, "nr_sv") == 0) {
			int n = model->nr_class;
			model->nSV = Malloc(int, n);
			for (int i = 0; i < n; i++)
				fscanf(fp, "%d", &model->nSV[i]);
		} else if (strcmp(cmd, "SV") == 0) {
			while (1) {
				int c = getc(fp);
				if (c == EOF || c == '\n') break;
			}
			break;
		} else {
			fprintf(stderr, "unknown text in model file: [%s]\n", cmd);

			setlocale(LC_ALL, old_locale);
			free(old_locale);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	char *p, *endptr, *idx, *val;

	while (readline(fp) != NULL) {
		p = strtok(line, ":");
		while (1) {
			p = strtok(NULL, ":");
			if (p == NULL)
				break;
			++elements;
		}
	}
	elements += model->count_instance;

	fseek(fp, pos, SEEK_SET);

	int m = model->nr_class - 1;
	int count_instance = model->count_instance;
	model->sv_coef = Malloc(double *, m);
	int i;
	for (i = 0; i<m; i++)
		model->sv_coef[i] = Malloc(double, count_instance);
	model->SV = Malloc(svm_node*, count_instance);
	svm_node *x_space = NULL;
	if (count_instance>0) x_space = Malloc(svm_node, elements);

	int j = 0;
	for (i = 0; i < count_instance; i++) {
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p, &endptr);
		for (int k = 1; k < m; k++) {
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p, &endptr);
		}

		while (1) {
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL) {
				break;
			}
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			x_space[j].value = strtod(val, &endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr) {
	if (model_ptr->free_sv && model_ptr->count_instance > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if (model_ptr->sv_coef) {
		for (int i = 0; i < model_ptr->nr_class - 1; i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label = NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB = NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr) {
	if (model_ptr_ptr != NULL && *model_ptr_ptr != NULL) {
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param) {
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param) {
	// svm_type

	int svm_type = param->svm_type;
	if (svm_type != C_SVC &&
		svm_type != NU_SVC &&
		svm_type != ONE_CLASS &&
		svm_type != EPSILON_SVR &&
		svm_type != NU_SVR)
		return "unknown svm type";

	// kernel_type, degree

	int kernel_type = param->kernel_type;
	if (kernel_type != LINEAR &&
		kernel_type != POLY &&
		kernel_type != RBF &&
		kernel_type != SIGMOID &&
		kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if (param->gamma < 0)
		return "gamma < 0";

	if (param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if (param->cache_size <= 0)
		return "cache_size <= 0";

	if (param->eps <= 0)
		return "eps <= 0";

	if (svm_type == C_SVC ||
		svm_type == EPSILON_SVR ||
		svm_type == NU_SVR)
	if (param->C <= 0)
		return "C <= 0";

	if (svm_type == NU_SVC ||
		svm_type == ONE_CLASS ||
		svm_type == NU_SVR)
	if (param->nu <= 0 || param->nu > 1)
		return "nu <= 0 or nu > 1";

	if (svm_type == EPSILON_SVR)
	if (param->p < 0)
		return "p < 0";

	if (param->shrinking != 0 &&
		param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if (param->probability != 0 &&
		param->probability != 1)
		return "probability != 0 and probability != 1";

	if (param->probability == 1 &&
		svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible

	if (svm_type == NU_SVC) {
		int count_instance = prob->count_instance;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int, max_nr_class);
		int *count = Malloc(int, max_nr_class);

		int i;
		for (i = 0; i < count_instance; i++) {
			int this_label = (int)prob->class_value[i];
			int j;
			for (j = 0; j < nr_class; j++)
			if (this_label == label[j]) {
				++count[j];
				break;
			}
			if (j == nr_class) {
				if (nr_class == max_nr_class) {
					max_nr_class *= 2;
					label = (int *)realloc(label, max_nr_class*sizeof(int));
					count = (int *)realloc(count, max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		for (i = 0; i < nr_class; i++) {
			int n1 = count[i];
			for (int j = i + 1; j<nr_class; j++) {
				int n2 = count[j];
				if (param->nu*(n1 + n2) / 2 > min_psvm(n1, n2)) {
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model) {
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA != NULL && model->probB != NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		model->probA != NULL);
}

void svm_set_print_string_function(void(*print_func)(const char *)) {
	if (print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}
