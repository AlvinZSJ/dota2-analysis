################
# This module gathers classes and methods for FP-tree and conditional FP-tree construction
#
# Usage: FPTree.py [-g | -p] [-s min_support] [-h min_height] [-i input_csv] [-o output_csv]
#
#        To generate frequent itemsets of input data <input csv>, saving to <output csv>, use commands:
#        python3 FPTree.py -g -s min_support -i <input csv> -o <output csv>
#
#        To print all conditional FP-trees generated from input data with height >= min_height, use commands:
#        python3 FPTree.py -p -s min_support -h min_height -i <input csv>
#
# Author: ZHANG Shenjia
###############


import csv
import time
import sys


def read_csv(input_csv):
    """
    read data from csv file

    :param input_csv: name of csv file which hold the data
    :return: (list of lists) data read from csv file
    """
    print("====================\n"
          "| Read from input csv...")
    data = []
    cTime = time.time()
    with open(input_csv, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            # Remove empty item
            line_data = [item for item in line if item != ""]
            data.append(line_data)

    print("--------------------\n"
          "| Time: {}s".format(round(time.time() - cTime, 4)))
    return data


def write_csv(data, output_csv):
    """
    Write data to target csv file

    :param data: data to be saved
    :param output_csv: target csv file to save data
    """
    cTime = time.time()
    print("====================\n"
          "| Saving frequent itemsets to \033[91m {}\033[00m".format(output_csv))

    # Save frequent itemsets into csv file
    with open(output_csv, "w") as f:
        for itemset in data:
            # Set the itemsets format to be "{item1, item2, ...}"
            freq_itemsets = "\"{" + str(itemset).strip("[").strip("]").replace("'", "") + "}\"\n"
            f.write(freq_itemsets)

    print("--------------------\n"
          "| Time: {}s\n"
          "====================".format(round(time.time() - cTime, 4)))


class Node:
    """
    Node class for FP-tree and conditional FP-tree construction

    Attributes
    ----------
    parent:         (Node) parent node of the current node
    item:           (str) the item on the current node
    freq:           (int) the frequent count of item on this node
    depth:          (int) the depth from root to this node
    children:       (List of Nodes) children nodes of this node
    node_link:      (Node) the link to the node with same item in FP-tree
    curr_structure: (List) structure of current and all sub nodes

    """
    def __init__(self, parent, item, freq=1, depth=1):
        """
        Initialize attributes of a Node
        :param parent: (Node) the parent node of this node
        :param item: (str) item on current node
        :param freq: (int) the frequent count of item on this node
        :param depth: (int) the depth from root to this node
        """
        self.freq = freq
        self.item = item
        self.depth = depth
        self.parent = parent

        # children nodes of this node
        self.children = []
        # the link to the node with same item in FP-tree
        self.node_link = None
        # structure of current and all sub nodes
        self.curr_structure = []

    def search_child(self, item):
        """
        Search the node's children for the node which contain the input item
        :param item: item to be searched
        :return: if the node containing the input item exists,
                 return it. Otherwise return None
        """
        for child in self.children:
            if item == child.item:
                return child

        return None

    def get_link_node(self):

        return self.node_link

    def get_parent_node(self):

        return self.parent

    def add_child(self, child):

        self.children.append(child)

    def collect_structure(self, sub_structure):

        self.curr_structure.append(sub_structure)

    def is_leaf(self):

        if not self.children:
            return True
        else:
            return False


class HeaderNode(Node):
    """
    Header node stored in header

    """
    def __init__(self, parent, item, freq=1, depth=1, last_node=None):
        Node.__init__(self, parent, item, freq, depth)
        self.last_node = last_node

    def get_last_node(self):
        return self.last_node

    def set_last_node(self, last_node):
        self.last_node = last_node


class FPTree:
    """
    Class for FP-tree

    Build the FP-tree using whole transactions to generate frequent itemsets.
    1. Deduce the ordered frequent items.
       Scan the transactions to count all items,
       find frequent items which support >= min support, then remove the infrequent items.

       Sort the frequent items according to DESC order of support count,
       if the support count is equal, sort items based on alphabetic order

    2. Construct FP-tree
       For each filtered transaction, insert the sorted items to FP-tree one by one.
       If the following item is contained in current node's children nodes,
       increase the target child node's frequency by 1,
       else create a new child containing the item, and set its frequency to 1.

       Each time build a new node, link it to the previous node which is same to the current one.
       Finally use header table to store first node of each unique node for future searching.

    3. Construct conditional FP-trees
       Build all conditional FP-trees on condition item, including sub problems.
       For candidates like \["e", "d", "c", "b", "a"], given a condition item, e.g. "e",
       first use header to find all nodes containing item "e",
       then use bottom up search to find all items on the path with suffix "e".
       count found items and filter them to get frequent items.

       Then choose each item in the frequent items to form conditions for sub problems, e.g. "ae", "be", "ce", etc.
       build the conditional tree recursively util no more frequent items can be used to form sub conditions.
       Finally all sub conditions are tried and all frequent itemsets are generated on condition "e".

    4. Determine the frequent patterns
        For each candidate like "e", build conditional tree recursively and find the entire frequent itemsets.


       Attributes
       ----------
       min_support : (int) the minimum support of the frequent itemsets

       root :        (Node) the root node of the FP-tree

       item_freq :   (dict) the dict of frequent itemsets
                     form: {item: frequency count}

       header:       (dict) header table
                     form: {item: first node contain item in FP-tree}
    """

    def __init__(self, transactions, min_support_count):
        """
        Initialize the FP-tree

        :param transactions: input transactions
        :param min_support:  the minimum support to build a FP-tree
        """
        self.min_support = min_support_count

        # initialize the root of FP-tree
        self.root = Node(None, None)

        # initialize frequent itemsets to None
        self.item_freq = None

        # get header of FP-tree
        self.header = self.fp_growth(transactions)

    def fp_growth(self, transactions):
        """
        Use transactions to build a FP-tree

        :param transactions: input set of transactions
        :return: header table of the built FP-tree
        """
        root = self.root

        # calculate the frequency of each item and construct a frequent itemsets dict
        self.item_freq = self.cal_item_freq(transactions)

        # initialize a empty header table for each unique item in the transactions
        header = self.build_header(self.item_freq)

        # use each transaction to build the FP-tree
        for transaction in transactions:

            # process transaction, including removing infrequent items and sorting
            transaction_freq_items = \
                self.process_transaction(transaction, self.item_freq)

            header = \
                self.insert_tree(transaction_freq_items, root, header)

        return header

    @staticmethod
    def build_header(item_freq):
        """
        Initialize a empty header for all frequent items after firstly scan the transactions

        :param item_freq: (dict) {item: frequent count of the item}
                          frequent count of the frequent items with support > minimum support

        :return: an empty header
        """
        header = {}

        # insert each frequent item as key to the header
        for item in item_freq:
            header[item] = None

        return header

    @staticmethod
    def insert_tree(transaction, root, header, freq=1):
        """
        Insert transaction into FP-tree / conditional FP-tree with input root and header,
        and the frequent count of each item in the transaction is set to input parameter freq

        For each item in a transaction, if it is contained in one of the children nodes of current node,
        increase the frequent count of item in that child node by input freq.
        Else create a new node containing the item and add it to current node's children nodes

        :param transaction: transaction which will be insert to FP-tree / conditional FP-tree

        :param root:        root of the FP-tree / conditional FP-tree

        :param header:      header of the FP-tree / conditional FP-tree

        :param freq:        frequent count of each item in the transaction

        :return: the updated header of the FP-tree / conditional FP-tree
        """
        curr_node = root

        for item in transaction:

            # Search if the item is contained in one of the current node's children nodes
            node_contain_item = curr_node.search_child(item)

            # If the node is found, increase the frequent count by input count "freq"
            if node_contain_item is not None:
                node_contain_item.freq += freq
                curr_node = node_contain_item

            else:

                # Update header
                if header[item] is None:
                    # if the node has not been found, create a new node with current node as parent node
                    new_child = \
                        HeaderNode(curr_node, item, freq=freq, depth=curr_node.depth + 1, last_node=None)
                    new_child.set_last_node(new_child)

                    # If the header is empty for the item, add the new child to the corresponding position
                    header[item] = new_child

                else:
                    new_child = Node(curr_node, item, freq=freq, depth=curr_node.depth + 1)

                    header_node = header[item]
                    last_node = header_node.get_last_node()
                    last_node.node_link = new_child
                    header_node.set_last_node(new_child)

                    # # Search for the last node containing the item
                    # while curr_head.node_link is not None:
                    #     curr_head = curr_head.node_link
                    #
                    # curr_head.node_link = new_child

                curr_node.add_child(new_child)
                curr_node = new_child

        return header

    @staticmethod
    def process_transaction(transaction, item_freq):
        """
        process transaction which will be used to construct FP-tree / conditional FP-tree,
        including removing infrequent items and sorting

        :param transaction: transaction containing items

        :param item_freq: (dict) {item: the frequent count of the item}
                          frequent items and their frequent count

        :return: frequent items in the input transaction
        """
        # remove infrequent items
        freq_items = \
            [item for item in transaction if item in item_freq]

        # if there is 0 / 1 frequent item, return it
        if len(freq_items) <= 1:
            return freq_items

        # sort the items according to their frequent count in DESC order first,
        # and for items with same frequent count, sort them in alphabetic order
        freq_items = \
            sorted(freq_items, key=lambda x: (-item_freq[x], x))

        return freq_items

    def build_cond_fp(self, cond_items, cond_header, min_height=0, print_structure=False):
        """
        Build all conditional FP-trees on condition items recursively, including sub problems.

        For a transaction, e.g. ["a", "b", "c", "d", "e"], given a condition item, e.g. "e",
        the corresponding input "cond_items" is ["e"].

        First use header of FP-tree / conditional FP-tree "cond_header" to find all nodes containing item "e",
        then use bottom up search to find all items on the path ending with suffix "e".
        All item frequent count in the same branch will be set to the same value, which is the frequent count of "e"
        count same items and filter them to get frequent itemsets.

        Then choose each item in the frequent items to form conditions for sub problems, e.g. "ae", "be", "ce", etc.
        build the conditional tree recursively util no more frequent items can be used to form sub conditions.
        Finally all sub conditions are tried and all frequent itemsets are generated on condition "e".
        For each candidate like "e", build conditional tree recursively and find the entire frequent itemsets.

        :param cond_items:       (list) conditional items, ranking in priority of condition
                                 e.g. for condition "ae", it is ["e", "a"]

        :param cond_header:      (dict) header of FP-tree / conditional FP-tree.
                                 At first, it is header of FP-tree
                                 In recursion, it is header of conditional FP-tree

        :param min_height:       (int) Minimum height of conditional FP-trees

        :param print_structure:  (boolean) if True, print conditional FP-tree structure in each recursion

        :return: Generated frequent itemsets until current recursion
        """
        # get condition item used in current recursion from conditional items
        curr_cond_item = cond_items[-1]

        # Get conditional transactions and their supports, and items' frequent count
        cond_transcations_info, item_freq = \
            self.get_cond_itemsets(cond_header, curr_cond_item)

        # Filter items to remove infrequent items
        item_freq = self.filter_item(item_freq)

        # Initialize the root of conditional FP-tree
        root = Node(None, "Null Set")

        # Create conditional FP-tree on conditions of current recursion
        curr_cond_header = self.build_header(item_freq)
        for info in cond_transcations_info:

            # remove infrequent items and sort
            processed_transcation = \
                self.process_transaction(info["cond_transaction"],
                                         item_freq)

            curr_cond_header = \
                self.insert_tree(transaction=processed_transcation,
                                 root=root,
                                 header=curr_cond_header,
                                 freq=info["support"])

        # The condition items are determined to be frequent itemsets
        curr_freq_itemsets = [cond_items]

        # Stop recursion if the height of conditional FP-tree is smaller than min height
        cond_tree_height = self.get_tree_height(root)
        if cond_tree_height < min_height:
            return curr_freq_itemsets

        # See if the conditional tree structure should be print out
        if print_structure:
            cond_tree_structure = self.get_tree_structure(root)
            print(cond_tree_structure)

        # Use each frequent item on current conditions to form conditions of sub problems
        for item in list(item_freq):
            cond_items_copy = cond_items.copy()
            cond_items_copy.append(item)

            # Use the newly formed conditions to build new conditional FP-trees recursively
            freq_itemsets = self.build_cond_fp(cond_items=cond_items_copy,
                                               cond_header=curr_cond_header,
                                               min_height=min_height,
                                               print_structure=print_structure)

            curr_freq_itemsets.extend(freq_itemsets)

        return curr_freq_itemsets

    def gen_freq_itemsets(self, output_csv=None):
        """
        Generate frequent itemsets by building all conditional FP-trees

        :return: frequent itemsets
        """
        freq_itemsets_info = []

        cTime = time.time()
        print("====================\n"
              "| Generating frequent itemsets with support count >= "
              "\033[91m {}\033[00m...".format(self.min_support))

        for item in list(self.item_freq):

            # Use each frequent item in FP-tree to build conditional FP-trees
            freq_cond_itemsets = self.build_cond_fp(cond_items=[item],
                                                    cond_header=self.header,
                                                    min_height=0,
                                                    print_structure=False)

            freq_itemsets_info.extend(freq_cond_itemsets)

        print("--------------------\n"
              "| Time: {}s".format(round(time.time() - cTime, 4)))

        # save frequent itemsets to target csv file
        if output_csv is not None:
            write_csv(freq_itemsets_info, output_csv)

        return freq_itemsets_info

    def print_cond_trees(self, min_height):
        """
        Print all conditional FP-trees with height >= min height

        :param min_height: min height of conditional FP-trees
        """
        print("====================\n"
              "| Structures of conditional FP-trees with height >= "
              "\033[91m {}\033[00m are:\n"
              "--------------------".format(min_height))

        for item in list(self.item_freq):
            self.build_cond_fp(cond_items=[item],
                               cond_header=self.header,
                               min_height=min_height,
                               print_structure=True)

    def get_tree_structure(self, root):
        """
        Use DFS to get the structure of the tree with input root in a nested list format

        :param root: root of the tree
        :return: structure of sub trees of root
        """
        return self.get_subtree_structure(root)

    def get_subtree_structure(self, curr_node):
        """
        Get structure of sub trees of current node

        :param curr_node: current node
        :return: structure of sub trees of current node in a format of nested lists
        """
        # Get structure of current node
        curr_node.collect_structure("{}  {}".format(curr_node.item, curr_node.freq))

        # Stop recursion if current node is a leaf node
        if curr_node.is_leaf():
            return curr_node.curr_structure

        children = curr_node.children

        # If the current node is not a splitting node, just go down through the path
        if len(children) == 1:

            branch_sub_structure = self.get_subtree_structure(children[0])
            curr_node.collect_structure(branch_sub_structure)

        else:

            sub_structures = []

            # If current node is a splitting node, go down through each path
            for child in curr_node.children:
                branch_sub_structure = self.get_subtree_structure(child)

                if len(branch_sub_structure) == 1:

                    # If one of the children nodes of the splitting node is a leaf node,
                    # directly add it to structure of current level
                    sub_structures.extend(branch_sub_structure)

                else:

                    # collect all sub structures separately
                    sub_structures.append(branch_sub_structure)

            curr_node.collect_structure(sub_structures)

        return curr_node.curr_structure

    def get_cond_itemsets(self, cond_header, cond_item):
        """
        Use header of conditional FP-tree to find all itemsets on condition of cond_item

        :param cond_header: (dict) header of conditional FP-tree
                            In recursion, it is the header of previous conditional tree.
                            e.g. transaction is sth like ["a", "b", "c", "e"] and condition items are ["e", "a"].
                            In recursion, current cond_item is "a", cond_header is the header of conditional FP-tree
                            which condition is "e".

        :param cond_item:   condition item

        :return: cond_transcations_info: (dict) info of transactions on condition "cond_item".
                                         Form: {"cond_transaction": conditional transaction,
                                                "support": support of this transaction}

                 item_freq:              (dict) {item: frequent count of the item}
        """
        # Get first node containing the item in conditional tree with header: "cond_header"
        cond_node = cond_header[cond_item]

        item_freq = {}
        cond_transcations_info = []

        # Traverse each conditional node on the chain
        while cond_node is not None:
            curr_node = cond_node.parent
            max_branch_freq = cond_node.freq
            cond_transcation = []

            # Use bottom up search to find all items on path ending with node containing the condition item
            while curr_node.parent is not None:
                # Get itemsets on condition of item
                item_freq = self.upsert_item(item=curr_node.item,
                                             count=max_branch_freq,
                                             item_freq=item_freq)

                cond_transcation.append(curr_node.item)

                # Bottom up
                curr_node = curr_node.parent

            # Shift current node to next condition node on the chain
            cond_node = cond_node.node_link

            # Gather info of conditional transaction
            cond_transcations_info.append({"cond_transaction": cond_transcation,
                                           "support": max_branch_freq})

        return cond_transcations_info, item_freq

    def get_tree_height(self, root):
        """
        Recursively go down from the root and use DFS to get max height of leaf nodes as tree height

        :param root: root of the tree
        :return: height of the tree
        """
        return self.get_subtree_height(root)

    def get_subtree_height(self, node, max_height=0):
        """
        Get height of the tree with the input root

        :param node: current node
        :param max_height: max height found util current node
        :return: max height after going through this node
        """
        # If current node is a leaf node, return max height
        if not node.children:
            return max(max_height, node.depth)

        # Recursively go down children nodes to get max height
        for child in node.children:
            max_height = \
                max(max_height, self.get_subtree_height(child, max_height))

        return max_height

    def filter_item(self, item_freq):
        """
        Filter items to remove infrequent items

        :param item_freq: (dict) {item: frequent count of the item}
        :return: filtered item_freq
        """
        # use list to avoid the error: "dict change size during iteration"
        for item in list(item_freq):

            # If the frequent count is smaller than min support count, remove the item
            if item_freq[item] < self.min_support:
                item_freq.pop(item)

        return item_freq

    def cal_item_freq(self, transactions):
        """
        Calculate the frequent count of each item in transactions

        :param transactions: (list of list) input transactions
        :return: (dict) {item: frequent count of the item}
        """
        item_freq = {}

        for transaction in transactions:

            for item in transaction:
                item_freq = self.upsert_item(item, 1, item_freq)

        item_freq = self.filter_item(item_freq)

        return item_freq

    @staticmethod
    def upsert_item(item, count, item_freq):
        """
        If item has already in frequent items dict, update its frequency.
        Else add the item into the frequent item dict and set its frequency to count

        :param item: item used to update frequent items dict
        :param count: the frequency of input item
        :param item_freq: the frequent items dict
        :return: the updated frequent items dict
        """
        if item in item_freq:
            item_freq[item] += count
        else:
            item_freq[item] = count

        return item_freq


if __name__ == "__main__":
    # help message
    help = "usage: FPTree.py [-g | -p] [-s min_support] [-h min_height] [-i input_csv] [-o output_csv]\n" \
           "\n" \
           "To generate frequent itemsets of input data <input csv>, saving to <output csv>, use commands:\n" \
           "python3 FPTree.py -g -s min_support -i <input csv> -o <output csv>\n" \
           "\n" \
           "To print all conditional FP-trees generated from input data with height >= min_height, use commands:\n" \
           "python3 FPTree.py -p -s min_support -h min_height -i <input csv> \n" \
           "\n" \
           "optional arguments:\n" \
           "  --help                show this help message and exit\n"\
           "  -g                    Generate frequent itemsets\n" \
           "  -p                    Print all conditional FP-trees\n"\
           "  -s min_support        set the min support of frequent itemsets\n"\
           "  -h min_height         set the min height of conditional FP-tres\n"\
           "  -i input_csv          set input csv\n"\
           "  -o output_csv         set output csv\n"

    # if "--help" in command, print help message
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print(help)

    # if "-p" or "-g" in commands, be ready to read
    elif "-p" in sys.argv or "-g" in sys.argv:

        try:
            # Read min support and data file in command line
            min_support = int(sys.argv[sys.argv.index("-s") + 1])
            transaction_file = sys.argv[sys.argv.index("-i") + 1]

        except:
            print("Your command cause error. Please use following "
                  "command correctly: \n{}".format(help))
            exit(1)

        # Read data from input csv file
        transactions = read_csv(transaction_file)

        cTime = time.time()
        print("====================\n"
              "| Building FP-tree...")

        # Initialize a FP-tree
        fp_tree = FPTree(transactions, min_support)
        print("--------------------\n"
              "| Time: {}s".format(round(time.time() - cTime, 4)))

        if "-g" in sys.argv:

            # If "-g" in command, set the output csv file
            output_csv = sys.argv[sys.argv.index("-o") + 1]

            # Generate frequent itemsets and save them to the file
            freq_itemsets_info = fp_tree.gen_freq_itemsets(output_csv)

        if "-p" in sys.argv:

            # If "-p" in command, set min height of conditional FP-trees
            min_height = int(sys.argv[sys.argv.index("-h") + 1])

            cTime = time.time()

            # Print all conditional FP-trees
            fp_tree.print_cond_trees(min_height)
            print("--------------------\n"
                  "| Time: {}s\n"
                  "====================".format(round(time.time() - cTime, 4)))

    # If the FP-tree.py script is run without command or run directly from IDE,
    # use default min support and min height to generate frequent itemsets and conditional FP-trees
    elif len(sys.argv) == 1:
        min_support = 10
        min_height = 2
        transaction_file = "radiant_win_radiant_heros.csv"
        output_csv = "radiant_win_radiant_heros_freq.csv"

        # Read data from input csv file
        transactions = read_csv(transaction_file)

        cTime = time.time()
        print("====================\n"
              "| Building FP-tree...")

        # Initialize a FP-tree
        fp_tree = FPTree(transactions, min_support)
        print("--------------------\n"
              "| Time: {}s".format(round(time.time() - cTime, 4)))

        # generate frequent itemsets and save them to the output csv file
        freq_itemsets_info = fp_tree.gen_freq_itemsets(output_csv)


        cTime = time.time()
        # Print all conditional FP-trees
        fp_tree.print_cond_trees(min_height)
        print("--------------------\n"
              "| Time: {}s\n"
              "====================".format(round(time.time() - cTime, 4)))

    else:
        print("Please use correct input format: \n{}".format(help))