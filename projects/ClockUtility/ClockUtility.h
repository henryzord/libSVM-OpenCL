#ifndef _CLOCK_UTILITY_H
#define _CLOCK_UTILITY_H

#include <chrono>
#include <iostream>
#include <string>

using namespace std;

//get time elapsed since beginning of construction
std::chrono::high_resolution_clock::time_point clock_begin;

string new_path, filename;

void printTimeElapsed(char *pre_message, char *pos_message) {
	std::chrono::high_resolution_clock::time_point clock_now = std::chrono::high_resolution_clock::now();
	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_now - clock_begin);
	cout << pre_message << ms.count() << " miliseconds" << pos_message;
}

void startClock() {
	clock_begin = std::chrono::high_resolution_clock::now();
}

//filename block
//directs stream from default output device to file
void startOutputCapture(int argc, char **argv, int min_params) {
	if(argc < min_params) {
		fprintf(stderr, "Less parameters than needed for output capture.\n");
		return;
	}

	string dataset_name(argv[1]), program_name(argv[0]);
	new_path = string(argv[2]);
	dataset_name = dataset_name.substr(dataset_name.find_last_of("\\") + string("\\").length(), dataset_name.find_last_of(".") - dataset_name.find_last_of("\\") - 1);
	program_name = program_name.substr(program_name.find_last_of("\\") + string("\\").length(), program_name.find_last_of(".") - program_name.find_last_of("\\") - 1);
	new_path = new_path.substr(0, new_path.find_last_of("\\") + string("\\").length());
	filename = program_name + string("_") + dataset_name + string(".txt");

	cout << "Output will be written in " << filename << " file at dataset folder." << endl;

	freopen(filename.c_str(), "w", stdout); //starts capturing output
}

void stopOutputCapture() {
	fclose(stdout); //stops capturing output
	remove((new_path + filename).c_str()); //remove previous file, if there's any
	rename(filename.c_str(), (new_path + filename).c_str()); //move file from .exe folder to dataset folder
}

#endif //_CLOCK_UTILITY_H