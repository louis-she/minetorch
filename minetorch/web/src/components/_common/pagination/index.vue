<template>
  <el-pagination
    :current-page="curPage"
    :page-sizes="[10, 20, 50]"
    :page-size="curPageSize"
    :total="totalNum"
    style="margin-top: 10px;text-align: right"
    layout="total, sizes, prev, pager, next, jumper"
    @size-change="handleSizeChange"
    @current-change="handleCurrentChange"/>
</template>

<script>
export default {
  props: {
    currentPage: {
      type: Number,
      default: 1
    },
    pageSize: {
      type: Number,
      default: 10
    },
    total: {
      type: Number,
      default: 0
    }
  },
  data () {
    return {
      curPage: this.currentPage,
      curPageSize: this.pageSize,
      totalNum: this.total
    }
  },
  watch: {
    currentPage (val) {
      this.curPage = val
    },
    pageSize (val) {
      this.curPageSize = val
    },
    total (val) {
      this.totalNum = val
    }
  },
  methods: {
    handleSizeChange (val) {
      this.curPage = 1
      this.$emit('change', this.curPage, val)
    },
    handleCurrentChange (val) {
      this.$emit('change', val, this.curPageSize)
    }
  }
}
</script>

<style lang="scss"></style>
