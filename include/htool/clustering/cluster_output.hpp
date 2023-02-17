#ifndef HTOOL_CLUSTERING_CLUSTER_OUTPUT_HPP
#define HTOOL_CLUSTERING_CLUSTER_OUTPUT_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace htool {

template <typename CoordinatePrecision>
class Cluster;

template <typename CoordinatesPrecision>
void print(const Cluster<CoordinatesPrecision> &cluster) {
    preorder_tree_traversal(
        cluster,
        [](const Cluster<CoordinatesPrecision> &current_cluster) {
            const auto &permutation = current_cluster.get_permutation();
            std::cout << '[';
            for (std::vector<int>::const_iterator i = permutation.cbegin() + current_cluster.get_offset(); i != permutation.cbegin() + current_cluster.get_offset() + current_cluster.get_size(); ++i)
                std::cout << *i << ',';
            std::cout << "\b]" << std::endl;
            ;
        });
}

template <typename CoordinatesPrecision>
void save_cluster_tree(const Cluster<CoordinatesPrecision> &cluster, std::string filename) {
    // Cluster tree properties
    std::ofstream output_permutation(filename + "_cluster_tree_properties.csv");

    output_permutation << "minclustersize: " << cluster.get_minclustersize() << "\n";
    output_permutation << "maximal depth: " << cluster.get_maximal_depth() << "\n";
    output_permutation << "minimal depth: " << cluster.get_minimal_depth() << "\n";
    output_permutation << "permutation: ";

    const auto &permutation = cluster.get_permutation();
    for (int i = 0; i < permutation.size(); i++) {
        output_permutation << permutation[i];
        if (i != permutation.size() - 1) {
            output_permutation << ",";
        } else {
            output_permutation << "\n";
        }
    }
    output_permutation << "local permutation: " << cluster.is_permutation_local() << "\n";

    // Cluster tree
    std::ofstream output_tree(filename + "_cluster_tree.csv");
    std::vector<std::vector<std::string>> outputs(cluster.get_maximal_depth() + 1);

    preorder_tree_traversal(
        cluster,
        [&outputs](const Cluster<CoordinatesPrecision> &current_cluster) {
            std::vector<std::string> infos_str = {NbrToStr(current_cluster.get_children().size()),
                                                  NbrToStr(current_cluster.get_rank()),
                                                  NbrToStr(current_cluster.get_offset()),
                                                  NbrToStr(current_cluster.get_size()),
                                                  NbrToStr(current_cluster.get_radius()),
                                                  NbrToStr(current_cluster.get_counter()),
                                                  NbrToStr(((is_cluster_on_partition(current_cluster) ? 1 : 0)))};
            for (int p = 0; p < current_cluster.get_center().size(); p++) {
                infos_str.push_back(NbrToStr(current_cluster.get_center()[p]));
            }

            std::string infos = join("|", infos_str);
            outputs[current_cluster.get_depth()].push_back(infos);
        });

    for (int p = 0; p < outputs.size(); p++) {
        for (int i = 0; i < outputs[p].size(); ++i) {
            output_tree << outputs[p][i];
            if (i != outputs[p].size() - 1) {
                output_tree << ',';
            }
        }
        output_tree << "\n";
    }
}

template <typename CoordinatesPrecision>
Cluster<CoordinatesPrecision> read_cluster_tree(std::string file_cluster_tree_properties, std::string file_cluster_tree) {
    std::string line{};
    std::string delimiter     = ",";
    std::string sub_delimiter = "|";
    std::vector<std::string> splitted_string{};

    // Clusters informtation
    std::ifstream input_tree(file_cluster_tree);
    if (!input_tree) {
        throw std::logic_error("[Htool error] Cannot open file containing tree");
    }
    std::vector<std::vector<std::string>> outputs;
    int count = 0;
    while (std::getline(input_tree, line)) {
        outputs.push_back(split(line, delimiter));
        count++;
    }

    // Cluster root
    std::vector<int> counter_offset(outputs.size(), 0);
    std::vector<std::string> local_info_str = split(outputs[0][0], "|");

    int number_of_children      = std::stoi(local_info_str[0]);
    int rank                    = std::stoi(local_info_str[1]);
    int offset                  = std::stoi(local_info_str[2]);
    int size                    = std::stoi(local_info_str[3]);
    CoordinatesPrecision radius = StrToNbr<CoordinatesPrecision>(local_info_str[4]);
    int counter                 = std::stoi(local_info_str[5]);
    bool is_on_partition        = std::stoi(local_info_str[6]);
    int spatial_dimension       = local_info_str.size() - 7;
    std::vector<CoordinatesPrecision> center(spatial_dimension, 0);
    for (int p = 0; p < spatial_dimension; p++) {
        center[p] = StrToNbr<CoordinatesPrecision>(local_info_str[p + 7]);
    }

    Cluster<CoordinatesPrecision> root_cluster(radius, center, rank, offset, size);
    auto &permutation = root_cluster.get_permutation();

    // Cluster tree
    std::ifstream input_permutation(file_cluster_tree_properties);
    if (!input_permutation) {
        throw std::logic_error("[Htool error] Cannot open file containing permutation");
    }

    std::getline(input_permutation, line);
    splitted_string = split(line, " ");
    root_cluster.set_minclustersize(std::stoul(splitted_string.back()));
    std::getline(input_permutation, line);
    splitted_string = split(line, " ");
    root_cluster.set_maximal_depth(std::stoi(splitted_string.back()));
    std::getline(input_permutation, line);
    splitted_string = split(line, " ");
    root_cluster.set_minimal_depth(std::stoi(splitted_string.back()));

    std::getline(input_permutation, line);
    splitted_string = split(split(line, " ").back(), delimiter.c_str());
    for (int i = 0; i < splitted_string.size(); i++) {
        permutation[i] = std::stoi(splitted_string[i]);
    }

    std::getline(input_permutation, line);
    splitted_string = split(line, " ");
    root_cluster.set_is_permutation_local(std::stoi(splitted_string.back()));

    // Build cluster tree
    std::stack<std::pair<Cluster<CoordinatesPrecision> *, int>> stack;
    stack.push(std::pair<Cluster<CoordinatesPrecision> *, int>(&root_cluster, number_of_children));

    while (!stack.empty()) {
        Cluster<CoordinatesPrecision> *curr = stack.top().first;
        number_of_children                  = stack.top().second;
        std::vector<int> number_of_children_next(number_of_children, 0);
        stack.pop();

        // Creating sons
        std::vector<Cluster<CoordinatesPrecision> *> children;
        for (int p = 0; p < number_of_children; p++) {
            local_info_str = split(outputs[curr->get_depth() + 1][counter_offset[curr->get_depth() + 1] + p], "|");

            number_of_children_next[p] = std::stoi(local_info_str[0]);
            radius                     = StrToNbr<CoordinatesPrecision>(local_info_str[4]);
            for (int l = 0; l < spatial_dimension; l++) {
                center[l] = StrToNbr<CoordinatesPrecision>(local_info_str[7 + l]);
            }
            rank                       = std::stoi(local_info_str[1]);
            offset                     = std::stoi(local_info_str[2]);
            size                       = std::stoi(local_info_str[3]);
            counter                    = StrToNbr<CoordinatesPrecision>(local_info_str[5]);
            bool is_child_on_partition = StrToNbr<bool>(local_info_str[6]);
            children.emplace_back(curr->add_child(radius, center, rank, offset, size, counter, is_child_on_partition));
        }

        if (!curr->is_leaf()) {
            counter_offset[curr->get_depth() + 1] += number_of_children;
            for (int p = number_of_children - 1; p != -1; p--) {
                stack.push(std::pair<Cluster<CoordinatesPrecision> *, int>(children[p], number_of_children_next[p]));
            }
        }
    }

    return root_cluster;
}

template <typename CoordinatesPrecision>
void save_clustered_geometry(const Cluster<CoordinatesPrecision> &cluster_tree, int spatial_dimension, const CoordinatesPrecision *x0, std::string filename, const std::vector<int> &depths) {

    std::ofstream output(filename + ".csv");
    const auto &permutation = cluster_tree.get_permutation();
    std::vector<std::vector<int>> outputs(depths.size());
    std::vector<int> counters(depths.size(), 0);
    for (int p = 0; p < outputs.size(); p++) {
        outputs[p].resize(permutation.size());
    }

    // Permuted geometric points
    for (int d = 0; d < spatial_dimension; d++) {
        output << "x_" << d << ",";
        for (int i = 0; i < permutation.size(); ++i) {
            output << x0[spatial_dimension * permutation[i] + d];
            if (i != permutation.size() - 1) {
                output << ",";
            }
        }
        output << "\n";
    }

    preorder_tree_traversal(
        cluster_tree,
        [&outputs, &depths, &counters](const Cluster<CoordinatesPrecision> &current_cluster) {
            std::vector<int>::const_iterator it = std::find(depths.begin(), depths.end(), current_cluster.get_depth());

            if (it != depths.end()) {
                int index = std::distance(depths.begin(), it);
                std::fill_n(outputs[index].begin() + current_cluster.get_offset(), current_cluster.get_size(), counters[index]);
                counters[index] += 1;
            }
        });

    for (int p = 0; p < depths.size(); p++) {
        output << depths[p] << ",";

        for (int i = 0; i < outputs[p].size(); ++i) {
            output << outputs[p][i];
            if (i != outputs[p].size() - 1) {
                output << ',';
            }
        }
        output << "\n";
    }
}

} // namespace htool
#endif
